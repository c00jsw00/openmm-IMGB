#!/usr/bin/env python
"""
Analysis script for antimicrobial peptide simulation in an implicit membrane
Calculates:
1. Z-position of the peptide over time
2. Tilt angle of the peptide with respect to the membrane normal
3. Secondary structure content over the trajectory
4. Position of hydrophilic vs. hydrophobic residues relative to membrane
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mdtraj as md
from scipy.spatial import distance
import seaborn as sns

# Parameters
output_dir = "output"
trajectory_file = os.path.join(output_dir, "trajectory.dcd")
topology_file = "fixed_p2.pdb"
membrane_center = 0.0  # z-coordinate of membrane center (Å)
membrane_thickness = 25.0  # hydrophobic thickness (Å)

# Define hydrophobicity scale (Eisenberg consensus scale)
hydrophobicity = {
    'ALA': 0.620, 'ARG': -2.530, 'ASN': -0.780, 'ASP': -0.900,
    'CYS': 0.290, 'GLN': -0.850, 'GLU': -0.740, 'GLY': 0.480,
    'HIS': -0.400, 'ILE': 1.380, 'LEU': 1.060, 'LYS': -1.500,
    'MET': 0.640, 'PHE': 1.190, 'PRO': 0.120, 'SER': -0.180,
    'THR': -0.050, 'TRP': 0.810, 'TYR': 0.260, 'VAL': 1.080
}

def load_trajectory():
    """Load trajectory and topology files"""
    print(f"Loading trajectory from {trajectory_file}...")
    try:
        traj = md.load(trajectory_file, top=topology_file)
        print(f"Loaded trajectory with {traj.n_frames} frames and {traj.n_atoms} atoms")
        return traj
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None

def calculate_z_position(traj):
    """Calculate z-position of peptide center of mass over time"""
    print("Calculating z-position of peptide over time...")
    
    # Calculate center of mass for each frame
    peptide_indices = traj.topology.select("protein")
    peptide_masses = np.array([a.element.mass for a in traj.topology.atoms 
                              if a.index in peptide_indices])
    
    # Normalize masses
    peptide_masses = peptide_masses / np.sum(peptide_masses)
    
    # Calculate center of mass
    com_z = np.zeros(traj.n_frames)
    for i in range(traj.n_frames):
        # Extract only peptide atoms and their z-coordinates
        peptide_z = traj.xyz[i, peptide_indices, 2]
        com_z[i] = np.sum(peptide_z * peptide_masses)
    
    # Convert to Angstroms and relative to membrane center
    com_z = com_z * 10 - membrane_center
    
    # Create time array (in ns)
    time = np.arange(traj.n_frames) * traj.timestep / 1000
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, com_z)
    plt.axhline(membrane_thickness/2, color='r', linestyle='--', label='Membrane interface')
    plt.axhline(-membrane_thickness/2, color='r', linestyle='--')
    plt.fill_between(time, -membrane_thickness/2, membrane_thickness/2, color='gray', alpha=0.2)
    plt.xlabel('Time (ns)')
    plt.ylabel('Z-position (Å)')
    plt.title('Peptide Center of Mass Z-Position Relative to Membrane')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'z_position.png'), dpi=300)
    plt.close()
    
    return time, com_z

def calculate_tilt_angle(traj):
    """Calculate tilt angle of peptide with respect to membrane normal"""
    print("Calculating peptide tilt angle...")
    
    # Define membrane normal (z-axis)
    membrane_normal = np.array([0, 0, 1])
    
    # Get alpha-carbon indices
    ca_indices = traj.topology.select("name CA")
    
    # Calculate principal axis and tilt angle for each frame
    tilt_angles = np.zeros(traj.n_frames)
    
    for i in range(traj.n_frames):
        # Get CA coordinates
        ca_coords = traj.xyz[i, ca_indices, :]
        
        # Center the coordinates
        ca_coords_centered = ca_coords - np.mean(ca_coords, axis=0)
        
        # Calculate inertia tensor
        inertia = np.zeros((3, 3))
        for j in range(len(ca_coords_centered)):
            x, y, z = ca_coords_centered[j]
            inertia[0, 0] += y**2 + z**2
            inertia[1, 1] += x**2 + z**2
            inertia[2, 2] += x**2 + y**2
            inertia[0, 1] -= x * y
            inertia[0, 2] -= x * z
            inertia[1, 2] -= y * z
        
        inertia[1, 0] = inertia[0, 1]
        inertia[2, 0] = inertia[0, 2]
        inertia[2, 1] = inertia[1, 2]
        
        # Get eigenvectors (principal axes)
        eigvals, eigvecs = np.linalg.eigh(inertia)
        
        # Principal axis is the eigenvector with smallest eigenvalue
        principal_axis = eigvecs[:, np.argmin(eigvals)]
        
        # Calculate angle between principal axis and membrane normal
        cos_angle = np.dot(principal_axis, membrane_normal) / np.linalg.norm(principal_axis)
        angle = np.arccos(np.abs(cos_angle)) * 180 / np.pi
        tilt_angles[i] = angle
    
    # Plot
    time = np.arange(traj.n_frames) * traj.timestep / 1000
    plt.figure(figsize=(10, 6))
    plt.plot(time, tilt_angles)
    plt.xlabel('Time (ns)')
    plt.ylabel('Tilt Angle (degrees)')
    plt.title('Peptide Tilt Angle Relative to Membrane Normal')
    plt.savefig(os.path.join(output_dir, 'tilt_angle.png'), dpi=300)
    plt.close()
    
    return time, tilt_angles

def analyze_residue_positions(traj):
    """Analyze positions of residues relative to membrane"""
    print("Analyzing residue positions relative to membrane...")
    
    # Get CA indices and residue names
    topology = traj.topology
    residues = list(topology.residues)
    ca_indices = [r.atom('CA').index for r in residues if r.atom('CA')]
    residue_names = [r.name for r in residues if r.atom('CA')]
    
    # Calculate average z-position for each residue in last half of trajectory
    half_point = traj.n_frames // 2
    avg_z_positions = np.zeros(len(ca_indices))
    
    for i, ca_idx in enumerate(ca_indices):
        z_positions = traj.xyz[half_point:, ca_idx, 2] * 10  # Convert to Angstroms
        avg_z_positions[i] = np.mean(z_positions) - membrane_center
    
    # Get hydrophobicity for each residue
    hydrophobicity_values = [hydrophobicity.get(res, 0) for res in residue_names]
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Create a colormap from blue (hydrophilic) to red (hydrophobic)
    colors = plt.cm.RdBu_r(plt.Normalize(-2.5, 1.5)(hydrophobicity_values))
    
    plt.bar(range(len(ca_indices)), avg_z_positions, color=colors)
    plt.axhline(membrane_thickness/2, color='k', linestyle='--', label='Membrane interface')
    plt.axhline(-membrane_thickness/2, color='k', linestyle='--')
    plt.fill_between([-0.5, len(ca_indices)-0.5], -membrane_thickness/2, membrane_thickness/2, 
                    color='gray', alpha=0.2, label='Membrane')
    
    plt.xticks(range(len(ca_indices)), [f"{i+1}:{name}" for i, name in enumerate(residue_names)], 
               rotation=90, fontsize=8)
    plt.xlabel('Residue')
    plt.ylabel('Average Z-Position (Å)')
    plt.title('Average Residue Z-Positions Relative to Membrane\n(colors indicate hydrophobicity)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residue_positions.png'), dpi=300)
    plt.close()
    
    # Create a hydrophobicity vs. z-position scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(hydrophobicity_values, avg_z_positions, c=hydrophobicity_values, 
                cmap='RdBu_r', s=50, alpha=0.8)
    
    for i, res in enumerate(residue_names):
        plt.annotate(f"{i+1}:{res}", 
                    (hydrophobicity_values[i], avg_z_positions[i]),
                    xytext=(5, 0), textcoords='offset points', fontsize=8)
    
    plt.axhline(membrane_thickness/2, color='k', linestyle='--', label='Membrane interface')
    plt.axhline(-membrane_thickness/2, color='k', linestyle='--')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.5, label='Membrane center')
    plt.fill_between(plt.xlim(), -membrane_thickness/2, membrane_thickness/2, 
                    color='gray', alpha=0.1, label='Membrane')
    
    plt.colorbar(label='Hydrophobicity')
    plt.xlabel('Hydrophobicity')
    plt.ylabel('Average Z-Position (Å)')
    plt.title('Residue Hydrophobicity vs. Z-Position')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'hydrophobicity_vs_position.png'), dpi=300)
    plt.close()
    
    return residue_names, avg_z_positions, hydrophobicity_values

def calculate_secondary_structure(traj):
    """Calculate secondary structure content over time"""
    print("Calculating secondary structure content...")
    
    try:
        # Calculate DSSP secondary structure
        dssp = md.compute_dssp(traj)
        
        # Convert to numeric representation for heatmap
        # H: alpha helix, E: beta sheet, C: coil
        ss_numeric = np.zeros(dssp.shape)
        ss_numeric[dssp == 'H'] = 0  # Alpha helix
        ss_numeric[dssp == 'E'] = 1  # Beta sheet
        ss_numeric[dssp == 'C'] = 2  # Coil
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create custom colormap
        cmap = plt.cm.get_cmap('viridis', 3)
        
        # Calculate time in ns
        time = np.arange(traj.n_frames) * traj.timestep / 1000
        
        # Determine residue labels
        residue_labels = [f"{i+1}:{r.name}" for i, r in enumerate(traj.topology.residues)]
        
        # Plot heatmap
        plt.imshow(ss_numeric.T, aspect='auto', cmap=cmap, 
                   extent=[0, time[-1], len(residue_labels)-0.5, -0.5])
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cmap(0), label='Alpha Helix (H)'),
            Patch(facecolor=cmap(1), label='Beta Sheet (E)'),
            Patch(facecolor=cmap(2), label='Coil (C)'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.colorbar(ticks=[0, 1, 2], label='Secondary Structure')
        plt.xlabel('Time (ns)')
        plt.ylabel('Residue')
        plt.yticks(np.arange(len(residue_labels)), residue_labels)
        plt.title('Secondary Structure Evolution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'secondary_structure.png'), dpi=300)
        plt.close()
        
        # Calculate and plot secondary structure content percentage
        ss_counts = np.zeros((traj.n_frames, 3))
        for i in range(traj.n_frames):
            unique, counts = np.unique(dssp[i], return_counts=True)
            for j, ss in enumerate(unique):
                if ss == 'H':
                    ss_counts[i, 0] = counts[j]
                elif ss == 'E':
                    ss_counts[i, 1] = counts[j]
                elif ss == 'C':
                    ss_counts[i, 2] = counts[j]
        
        # Convert to percentages
        ss_percentages = ss_counts / np.sum(ss_counts[0]) * 100
        
        # Plot percentages
        plt.figure(figsize=(10, 6))
        plt.stackplot(time, ss_percentages.T, labels=['Alpha Helix', 'Beta Sheet', 'Coil'],
                     colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.xlabel('Time (ns)')
        plt.ylabel('Content (%)')
        plt.title('Secondary Structure Content Over Time')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(output_dir, 'ss_percentages.png'), dpi=300)
        plt.close()
        
        return dssp
        
    except Exception as e:
        print(f"Error calculating secondary structure: {e}")
        return None

def main():
    """Main analysis function"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load trajectory
    traj = load_trajectory()
    if traj is None:
        return
    
    # Run analyses
    time, z_positions = calculate_z_position(traj)
    time, tilt_angles = calculate_tilt_angle(traj)
    residue_names, avg_z_positions, hydrophobicity_values = analyze_residue_positions(traj)
    dssp = calculate_secondary_structure(traj)
    
    print("Analysis complete! Results saved to the output directory.")

if __name__ == "__main__":
    main()
