#!/usr/bin/env python
"""
Simulation of an antimicrobial peptide using OpenMM with CHARMM36 force field
and a CustomGBForce for the implicit membrane model
"""

import os
import numpy as np
from openmm import app
import openmm as mm
from openmm import unit
from sys import stdout

# Parameters
temperature = 300 * unit.kelvin
timestep = 2.0 * unit.femtoseconds
nsteps = 500000  # 1 ns simulation
reportInterval = 1000  # Report every 2 ps
saveTrajectoryInterval = 5000  # Save trajectory frames every 10 ps

# Membrane parameters
membrane_thickness = 30.0  # Angstroms
solvent_dielectric = 78.3
solute_dielectric = 1.0

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

print("Loading peptide structure...")
pdb = app.PDBFile('fixed_p2.pdb')

# Setup force field
print("Setting up force field...")
forcefield = app.ForceField('charmm36.xml')

# Configure cutoff parameters
nonbonded_cutoff = 1.2 * unit.nanometers

# Create system with consistent cutoff settings
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.CutoffNonPeriodic,
    nonbondedCutoff=nonbonded_cutoff,
    constraints=app.HBonds
)

# Get NonbondedForce for charge/LJ parameters
nonbonded_force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))

# Configure CustomGBForce with identical cutoff
print("Setting up custom GB force...")
custom = mm.CustomGBForce()
custom.setNonbondedMethod(mm.CustomGBForce.CutoffNonPeriodic)  # Explicit non-periodic cutoff
custom.setCutoffDistance(nonbonded_cutoff.value_in_unit(unit.nanometer))

# Define GB parameters and energy terms
custom.addPerParticleParameter("q")
custom.addPerParticleParameter("radius")
custom.addPerParticleParameter("scale")
custom.addGlobalParameter("thickness", membrane_thickness/10)
custom.addGlobalParameter("solventDielectric", solvent_dielectric)
custom.addGlobalParameter("soluteDielectric", solute_dielectric)

# Add computed values for the GB model
custom.addComputedValue("Imol", 
                       "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
                       "U=r+sr2;"
                       "C=2*(1/or1-1/L)*step(sr2-r-or1);"
                       "L=max(or1, D);"
                       "D=abs(r-sr2);"
                       "sr2 = scale2*or2;"
                       "or1 = radius1-0.009; or2 = radius2-0.009", 
                       mm.CustomGBForce.ParticlePairNoExclusions)

custom.addComputedValue("Imem", 
                       "(1/radius+2*log(2)/thickness)/(1+exp(7.2*(abs(z)+radius-0.5*thickness)))", 
                       mm.CustomGBForce.SingleParticle)

custom.addComputedValue("B", 
                       "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
                       "psi=max(Imol,Imem)*or; or=radius-0.009", 
                       mm.CustomGBForce.SingleParticle)

# Add energy terms
custom.addEnergyTerm("28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935456*(1/soluteDielectric-1/solventDielectric)*q^2/B", 
                     mm.CustomGBForce.SingleParticle)

custom.addEnergyTerm("-138.935456*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;"
                     "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", 
                     mm.CustomGBForce.ParticlePairNoExclusions)

# Set GB parameters based on element type
default_radius_scale = {
    'H': (0.12, 0.85),
    'C': (0.17, 0.72),
    'N': (0.155, 0.79),
    'O': (0.15, 0.85),
    'S': (0.18, 0.96),
    'P': (0.18, 0.86)
}

# Add particles to GB force
for i, atom in enumerate(pdb.topology.atoms()):
    charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
    element = atom.element.symbol
    radius, scale = default_radius_scale.get(element, (0.15, 0.8))
    custom.addParticle([charge.value_in_unit(unit.elementary_charge), radius, scale])

# Disable charges in NonbondedForce (keep LJ)
for i in range(nonbonded_force.getNumParticles()):
    charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
    nonbonded_force.setParticleParameters(i, 0*unit.elementary_charge, sigma, epsilon)

# Add forces to system
custom.setForceGroup(1)
system.addForce(custom)

# Configure CUDA platform
print("Creating CUDA simulation...")
platform = mm.Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}

# Create simulation
simulation = app.Simulation(
    pdb.topology,
    system,
    mm.LangevinMiddleIntegrator(temperature, 1.0/unit.picosecond, timestep),
    platform,
    properties
)

# Set positions and minimize
simulation.context.setPositions(pdb.positions)
print("Minimizing energy...")
simulation.minimizeEnergy()

# Set velocities and reporters
simulation.context.setVelocitiesToTemperature(temperature)
simulation.reporters.append(app.DCDReporter(os.path.join(output_dir, 'trajectory.dcd'), saveTrajectoryInterval))

# Corrected StateDataReporter without extraComputeEnergy
#simulation.reporters.append(app.StateDataReporter(stdout, reportInterval,
#    step=True, time=True, potentialEnergy=True, temperature=True, speed=True,
#    separator='\t'))

# Reporter for log file (output/log)
simulation.reporters.append(app.StateDataReporter(os.path.join(output_dir, 'log'), reportInterval,
    step=True, time=True, potentialEnergy=True, temperature=True, speed=True,
    separator='\t'))

# Custom reporter for GB energy
class GBEnergyReporter:
    def __init__(self, file, reportInterval):
        self._reportInterval = reportInterval
        self._out = open(file, 'w')
    
    def __del__(self):
        self._out.close()
    
    def describeNextReport(self, simulation):
        return (self._reportInterval, False, False, True, False, None)
    
    def report(self, simulation, state):
        gb_energy = simulation.context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
        self._out.write(f"{state.getTime().value_in_unit(unit.picosecond):.2f}\t{gb_energy.value_in_unit(unit.kilocalorie_per_mole):.4f}\n")
        self._out.flush()

simulation.reporters.append(GBEnergyReporter(os.path.join(output_dir, 'gb_energy.txt'), reportInterval))

# Run simulation
print(f"Running simulation for {nsteps*timestep.value_in_unit(unit.picoseconds):.2f} ps...")
simulation.step(nsteps)
print("Simulation complete!")

# Save final structure
state = simulation.context.getState(getPositions=True)
with open(os.path.join(output_dir, 'final.pdb'), 'w') as f:
    app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
