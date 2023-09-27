"""
Targted MD simulation in OpenMM from an inital structure 
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
# Openmmtools provides additional intergrators, from package conda-forge::openmmtools
# from openmmtools.integrators import BAOABIntegrator
from sys import stdout
from copy import deepcopy
import numpy as np

# Function to add backbone position restraints
# Used for the NVT 
def add_backbone_pos_res(system, positions, atoms, restraint_force):
  force = CustomExternalForce("k_res*periodicdistance(x, y, z, x0, y0, z0)^2")
  force_amount = restraint_force * kilocalories_per_mole/angstroms**2
  force.addGlobalParameter("k_res", force_amount)
  force.addPerParticleParameter("x0")
  force.addPerParticleParameter("y0")
  force.addPerParticleParameter("z0")
  for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
    if atom.name in  ('CA', 'C', 'N'):
      force.addParticle(i, atom_crd.value_in_unit(nanometers))
  pos_res_sys = deepcopy(system)
  pos_res_sys.addForce(force)
  return pos_res_sys

# Define platform for compute and set precision for calcuations
platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

# Load in the PDB strucure
# Needs to be capped... I should build fix at some point
pdb_state_A = PDBFile('state_A_capped.pdb')
pdb_state_B = PDBFile('state_B_capped_trimmed_ca.pdb')

# Specifiy the forcefield
# OpenMM 8.0 does not support amber19FF via the force field function however you can and SHOULD use a prmtop / inpcrd pair  
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

# Combine the molecular topology and the forcefield
modeller_A = Modeller(pdb_state_A.topology, pdb_state_A.positions)
modeller_B = Modeller(pdb_state_B.topology, pdb_state_B.positions)

#Creates solvent box around the protein with a box size of protein size + (padding^2+1)
modeller_A.addSolvent(forcefield, padding=1.2*nanometers)

# Setup refrence positions for the systems using the solvent from state A and the state B protein. We assume that chain A is protein. 
modeller_A_solvent = deepcopy(modeller_A)
chains = [r for r in modeller_A_solvent.topology.chains()]
modeller_A_solvent.delete([chains[0]])
modeller_B.add(modeller_A_solvent.topology, modeller_A_solvent.positions)

# System preperation to create simulation object, add backbone restraints, and pullign forces
# Pulling forces are switched off for the initial equlibriations
system = forcefield.createSystem(modeller_A.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometers, constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)
pos_res_sys = add_backbone_pos_res(system, modeller_A.positions, modeller_A.topology.atoms(), 100)

rmsd_cv = openmm.RMSDForce(modeller_B, pull_index)
energy_expression = f"(spring_constant/2)*max(0, RMSD-RMSDmax)^2"
# energy_expression = f"RMSD"
pull_force = openmm.CustomCVForce(energy_expression)
pull_force.addCollectiveVariable('RMSD', rmsd_cv)
pull_force.addGlobalParameter('RMSDmax', 0.4)
pull_force.addGlobalParameter("spring_constant", 0)
pull_index = system.addForce(restraint_force)
pull_sys = deepcopy(pos_res_sys)
pull_sys.addForce(pull_force)

# Create the integrator to use for advacing the equations of motion.
# The paramters set are temperature, friction coefficient, and timestep.
# Mid point Langevin integrator that relies on the current time and postion in the intergration step 
# This should not be used for MC/MD methods, goes boom
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
integrator.setConstraintTolerance(0.00001)

simulation = Simulation(modeller_A.topology, pull_sys, integrator)
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True, volume=True, density=True))

# Perform energy minimization on complex to relax any high energy contacts 
print("Minimizing energy...")
simulation.context.setPositions(modeller_A.positions)
simulation.minimizeEnergy(maxIterations=5000)

#Checkpoint
simulation.saveState(('min.state'))
simulation.saveCheckpoint(('min.chk'))

# Warm up the system after inital vellocities are applied with a NVT run to 300k with backbone constraints.
simulation.context.setVelocitiesToTemperature(5*kelvin)
print('Warming up the system...')

# Define inital temprature 
Temp = 5

# 2.4 ns or 1200000 steps - every 20000 steps raise the temperature by 5 K, ending at 300 K
mdsteps = 1200000
for i in range(60):
  simulation.step(int(mdsteps/60) )
  temperature = (Temp+(i*Temp))*kelvin 
  integrator.setTemperature(temperature)
  
#Checkpoint
simulation.saveState(('NVT_heating.state'))
simulation.saveCheckpoint(('NVT_heating.chk'))

#NPT equilibration over 10ns while reducing backbone constraints
mdsteps = 500000
barostat =  pull_sys.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
simulation.context.reinitialize(True)

print('Running NPT equilibration...')
for i in range(100):
  simulation.step(int(mdsteps/100))
  simulation.context.setParameter('k_res', (float(99.02-(i*0.98))*kilocalories_per_mole/angstroms**2))

# Checkpoint
simulation.saveState(('NPT_equilibration.state'))
simulation.saveCheckpoint(('NPT_equilibration.chk'))

# Load checkpoint for going into NPT production
simulation.loadCheckpoint(('NPT_equilibration.chk'))

# Reset step and time counters for the production run
eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
positions = eq_state.getPositions()
velocities = eq_state.getVelocities()
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
integrator.setConstraintTolerance(0.00001)

# Zero out preious constraint that was placed on the backbone and set new simulation contexts 
simulation.context.setParameter('k_res', 0.0*kilocalories_per_mole/angstroms**2)
simulation.context.setParameter('k_pull', 5.0*kilocalories_per_mole/angstroms**2)
simulation.context.setPositions(positions)
simulation.context.setVelocities(velocities)

# Append reporters
simulation.reporters.append(DCDReporter(('traj_02.dcd'), 10000))
simulation.reporters.append(StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(PDBReporter(('prod_02_snapshots.pdb'), reportInterval = 10000))

# NPT production simulation for 100ns
cycles = 500
print('Running Production...')

for i in range(cycles):
  simulation.step(5000000)

print('Done!')
simulation.saveState(('prod_01.state'))
simulation.saveCheckpoint(('prod_01.chk'))