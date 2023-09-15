"""
Targted MD simulation in OpenMM from an inital structure 
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
# Openmmtools provides additional intergrators, from package conda-forge::openmmtools
# from openmmtools.integrators import BAOABIntegrator
from sys import stdout

# Function to add backbone position restraints
# Used for the NVT 
def add_backbone_pos_res(system, positions, atoms, restraint_force):
  force = CustomExternalForce("k_res*periodicdistance(x, y, z, x0, y0, z0)^2")
  force_amount = restraint_force * kilocalories_per_mole/angstroms**2
  force.addGlobalParameter("k", force_amount)
  force.addPerParticleParameter("x0")
  force.addPerParticleParameter("y0")
  force.addPerParticleParameter("z0")
  for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
    if atom.name in  ('CA', 'C', 'N'):
      force.addParticle(i, atom_crd.value_in_unit(nanometers))
  posres_sys = deepcopy(system)
  posres_sys.addForce(force)
  return pos_res_sys

# Function to add the pulling force
# Used for the NPT production
def add_RMSD_TMD_pullforce(system, positions, atoms, pull_force):
  pull_force = CustomExternalForce("k_pull*periodicdistance(x, y, z, x0, y0, z0)^2")
  pull_force_amount = pull_force * kilocalories_per_mole/angstroms**2
  pull_force.addGlobalParameter("k_pull", pull_force_amount)
  pull_force.addPerParticleParameter("x0")
  pull_force.addPerParticleParameter("y0")
  pull_force.addPerParticleParameter("z0")
  for i, (atom_crd, atom) in enumerate(zip(positions, atoms)):
    if atom.name in  ('CA', 'C', 'N'):
      force.addParticle(i, atom_crd.value_in_unit(nanometers))
  pull_res_sys = deepcopy(system)
  pull_res_sys.addForce(pull_force)
  return pull_res_sys

# Define platform for compute and set precision for calcuations
platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

# Load in the PDB strucure
pdb_state_A = PDBFile('state_A.pdb')
pdb_state_B = PDBFile('state_B.pdb')

# Specifiy the forcefield
# OpenMM 8.0 does not support amber19FF via the force field function however you can and SHOULD use a prmtop / inpcrd pair  
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
# Combine the molecular topology and the forcefield
system = forcefield.createSystem(pdb_state_A.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
modeller_A = Modeller(pdb_state_A.topology, pdb_state_A.positions)
modeller_B = Modeller(pdb_state_B.topology, pdb_state_B.positions)

#Creates solvent box around the protein with a box size of protein size + (padding^2+1)
modeller_A.addSolvent(forcefield, padding=2.0*nanometers)

# System preperation to creat simulation object, add backbone restraints, and pullign forces
# Pulling forces are switched off for the initial equlibriations
system = forcefield.createSystem(modeller_A.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*nanometers, constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)
pos_res_sys = add_backbone_posres(system, modeller_A.positions, modeller_A.topology.atoms(), 100)
pull_res_sys = add_RMSD_TMD_pullforce(pos_res_sys, modeller_B.positions, modeller_A.topology.atoms(), 0)

# Create the integrator to use for advacing the equations of motion.
# The paramters set are temperature, friction coefficient, and timestep.
# Mid point Langevin integrator that relies on the current time and postion in the intergration step 
# This should not be used for MC/MD methods, goes boom
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
integrator.setConstraintTolerance(0.00001)

simulation = Simulation(modeller_A.topology, pull_res_sys, integrator)
simulation.reporters.append(StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, volume=True, density=True))

# Perform energy minimization on complex to relax any high energy contacts 
print("Minimizing energy...")
simulation.minimizeEnergy(maxIterations=5000)

#Checkpoint
simulation.saveState(('min.state').as_posix())
simulation.saveCheckpoint(('min.chk').as_posix())

# Warm up the system after inital verollicies are applied with a NVT run to 300k with backbone constraints.
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
simulation.saveState(('NVT_heating.state').as_posix())
simulation.saveCheckpoint(('NVT_heating.chk').as_posix())

#NPT equilibration over 10ns while reducing backbone constraints
mdsteps = 5000000
barostat =  pull_res_sys.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
simulation.context.reinitialize(True)

print('Running NPT equilibration...')
for i in range(100):
  simulation.step(int(mdsteps/100))
  simulation.context.setParameter('k', (float(99.02-(i*0.98))*kilocalories_per_mole/angstroms**2))

# Checkpoint
simulation.saveState(('NPT_equilibration.state').as_posix())
simulation.saveCheckpoint(('NPT_equilibration.chk').as_posix())

# Load checkpoint for going into NPT production
simulation.loadCheckpoint(('NPT_equilibration.chk').as_posix())

# Reset step and time counters for the production run
eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
positions = eq_state.getPositions()
velocities = eq_state.getVelocities()
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
integrator.setConstraintTolerance(0.00001)

simulation = Simulation(modeller_A.topology, pull_res_sys, integrator)
simulation.context.setPositions(positions)
simulation.context.setVelocities(velocities)

# Removes previous constraints and enables the pulling forces
simulation.context.setParameter('k_res', 0.0)
simulation.context.setParameter('k_pull', 100.0)

# Append reporters
simulation.reporters.append(DCDReporter(('traj_01.dcd').as_posix(), 1000))
simulation.reporters.append(StateDataReporter(stdout, 10000, step=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True, totalSteps=mdsteps, separator='\t'))
simulation.reporters.append(PDBReporter(('prod_01_snapshots.pdb').as_posix(), reportInterval = 10000))

# NPT production simulation for 100ns
mdsteps = 50000000
cycles = mdsteps/1000
print('Running Production...')

for i in range(cycles):
  simulation.step(1000)

print('Done!')
simulation.saveState(('prod_01.state').as_posix())
simulation.saveCheckpoint(('prod_01.chk').as_posix())
