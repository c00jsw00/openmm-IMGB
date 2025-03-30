from openmm import app, unit
from openmm import *
import numpy as np
import os
import time

# ----------------------------
# Step 1: Prepare input files and parameters
# ----------------------------
input_pdb = "fixed_p2.pdb"        # Input structure file
output_dir = "output2"            # Output directory
os.makedirs(output_dir, exist_ok=True)  # Automatically create directory

# ----------------------------
# Define membrane-related parameters
# ----------------------------
MEMBRANE_CORE = 15.0 * unit.angstrom    # Membrane core region thickness
ESTER_REGION = 20.0 * unit.angstrom     # Ester layer boundary
EPSILON_CORE = 2.0                      # Core dielectric constant
EPSILON_ESTER = 4.0                     # Ester layer dielectric constant
EPSILON_WATER = 80.0                    # Water phase dielectric constant
C0 = 1.0                                # Formula coefficient
C1 = 0.8
D = 0.5
E = 0.1

# ----------------------------
# Step 2: Create system and load force field
# ----------------------------
pdb = app.PDBFile(input_pdb)
forcefield = app.ForceField('charmm36.xml')

# Get a system for atom parameters
temp_system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
    rigidWater=True
)

# Find NonbondedForce to get parameters
nonbonded_force = None
for force in temp_system.getForces():
    if isinstance(force, NonbondedForce):
        nonbonded_force = force
        break

if nonbonded_force is None:
    raise ValueError("No NonbondedForce found in the system")

# Save parameters
parameters = []
for i in range(nonbonded_force.getNumParticles()):
    charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
    parameters.append((charge, sigma, epsilon))

# Recreate system, but keep standard forces
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
    rigidWater=True,
    removeCMMotion=True,
    hydrogenMass=None
)

# Remove any existing NonbondedForce
for i in reversed(range(system.getNumForces())):
    if isinstance(system.getForce(i), NonbondedForce):
        system.removeForce(i)

# ----------------------------
# Step 3: Add custom GB force
# ----------------------------
# Create custom GB force
gb_force = CustomGBForce()

# Add particle parameters
gb_force.addPerParticleParameter("charge")
gb_force.addPerParticleParameter("radius")
gb_force.addPerParticleParameter("z")

# Add first computed value (must be ParticlePair type)
gb_force.addComputedValue("d", "r", CustomGBForce.ParticlePair)

# Add single particle computed values
z_core = MEMBRANE_CORE.value_in_unit(unit.angstrom)
z_ester = ESTER_REGION.value_in_unit(unit.angstrom)
gb_force.addComputedValue("epsilon", f"""
    step({z_core} - abs(z)) * {EPSILON_CORE} +
    step(abs(z) - {z_core}) * step({z_ester} - abs(z)) * {EPSILON_ESTER} +
    step(abs(z) - {z_ester}) * {EPSILON_WATER}
""", CustomGBForce.SingleParticle)

# Add other computed values
gb_force.addComputedValue("A4", "1/(radius + 0.5)", CustomGBForce.SingleParticle)
gb_force.addComputedValue("A7", "0.2/(radius^2 + 0.1)", CustomGBForce.SingleParticle)

gb_force.addComputedValue("alpha", f"""
    1/({C0}*A4 + {C1}*(3*epsilon/(3*epsilon + 2*1.0))*A7 + {D} + {E}/(epsilon + 1))
""", CustomGBForce.SingleParticle)

# Add energy term
gb_force.addEnergyTerm(f"""
    -166 * (1/1.0 - 1/(0.5*(epsilon1 + epsilon2))) *
    charge1 * charge2 / sqrt(r^2 + alpha1*alpha2*exp(-r^2/(4*alpha1*alpha2)))
""", CustomGBForce.ParticlePair)

# Add particle parameters to GB force
for atom_index, atom in enumerate(pdb.topology.atoms()):
    charge, sigma, epsilon = parameters[atom_index]
    radius = sigma * 0.5  # Radius is half of sigma
    z_pos = pdb.positions[atom_index][2].value_in_unit(unit.angstrom)
    gb_force.addParticle([charge, radius, z_pos])

# Add GB force to system
gb_force.setForceGroup(1)  # Set to group 1 for separate energy analysis
system.addForce(gb_force)

# ----------------------------
# Step 4: Initialize simulator
# ----------------------------
integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# ----------------------------
# Configure output reporters
# ----------------------------
from openmm.app import DCDReporter, StateDataReporter

# Trajectory reporter
traj_path = os.path.join(output_dir, "trajectory.dcd")
simulation.reporters.append(DCDReporter(traj_path, 1000))

# Log reporter
log_path = os.path.join(output_dir, "simulation_log.csv")

# Ensure output directory exists and has write permission
if os.path.exists(log_path):
    try:
        # If file exists, try to delete it to avoid permission issues
        os.remove(log_path)
    except (OSError, PermissionError) as e:
        print(f"Warning: Cannot delete existing log file: {e}")
        # Use a new filename with timestamp
        log_path = os.path.join(output_dir, f"simulation_log_{int(time.time())}.csv")

# Create main StateDataReporter
simulation.reporters.append(
    StateDataReporter(
        log_path,
        1000,  # Report every 1000 steps
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        speed=True,
        separator=','
    )
)

# Define a function to calculate and record different force group energies
def append_energy_data(simulation):
    """Record energies from different force groups to a separate file"""
    step = simulation.currentStep
    
    # Get total energy
    state = simulation.context.getState(getEnergy=True)
    total_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    
    # Get DHDGB energy (force group 1)
    gb_state = simulation.context.getState(getEnergy=True, groups={1})
    gb_energy = gb_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    
    # Get bonded energy (force group 0)
    bonded_state = simulation.context.getState(getEnergy=True, groups={0})
    bonded_energy = bonded_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    
    # Record to file
    energy_path = os.path.join(output_dir, "energy_components.csv")
    write_header = not os.path.exists(energy_path)
    
    with open(energy_path, 'a') as f:
        if write_header:
            f.write("Step,Total Energy (kJ/mol),DHDGB Energy (kJ/mol),Bonded Energy (kJ/mol)\n")
        f.write(f"{step},{total_energy},{gb_energy},{bonded_energy}\n")

# Create a custom reporter class for energy decomposition
class EnergyDecompositionReporter:
    def __init__(self, reportInterval):
        self._reportInterval = reportInterval
        
    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, False)
        
    def report(self, simulation, state):
        append_energy_data(simulation)

# Add energy decomposition reporter
simulation.reporters.append(EnergyDecompositionReporter(1000))

# ----------------------------
# Step 5: Run simulation
# ----------------------------
print("Starting production run...")
simulation.step(50000)

# Get final energies
context = simulation.context
state = context.getState(getEnergy=True)
total_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

gb_energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
bonded_energy = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

print(f"""
Simulation completed! Output files saved in: {output_dir}
├── trajectory.dcd     # Trajectory file
├── simulation_log.csv # Log file
└── energy_components.csv # Energy decomposition file

Final energies:
- Total energy: {total_energy} kJ/mol
- DHDGB energy: {gb_energy} kJ/mol
- Bonded energy: {bonded_energy} kJ/mol
""")