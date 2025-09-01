import mdtraj as md
import numpy as np
from collections import defaultdict
import itertools

class FeaturesGenerator:
    """
    Class to generate a PLUMED features.dat file from an MDTraj trajectory (single frame reference structure)
    and a selection dictionary specifying the types of features to include (cartesian, distances, angles, dihedrals).
    
    The reference structure is used to compute positions for pair selection in distances (if cutoff is provided).
    
    Usage:
    generator = FeaturesGenerator(ref_structure='reference.pdb', selection_dict=your_dict)
    generator.generate(output_file='features.dat')
    """
    
    def __init__(self, ref_structure: str, selection_dict: dict):
        """
        Initialize with path to reference structure (e.g., PDB file) and selection dictionary.
        
        Args:
            ref_structure: Path to the reference PDB or other MDTraj-loadable file (provides topology and coordinates).
            selection_dict: Dictionary specifying features, e.g.,
                {
                    'cartesian': {'selection': 'backbone'},
                    'distances': {'selection': 'name CA', 'cutoff': 0.4, 'periodic': True, 'bonded': False},
                    'angles': {'selection': 'backbone', 'periodic': True, 'bonded': True},
                    'dihedrals': {'selection': 'backbone', 'periodic': True, 'bonded': True}
                }
        """
        self.traj = md.load(ref_structure)
        self.top = self.traj.top
        self.selection_dict = selection_dict
    
    def generate(self, output_file: str = 'features.dat'):
        """
        Generate the features.dat file with PLUMED commands for the specified features.
        
        Args:
            output_file: Path to output features.dat file.
        """
        lines = []
        
        # Cartesian positions
        if 'cartesian' in self.selection_dict:
            sel = self.selection_dict['cartesian']['selection']
            atoms = self.top.select(sel)
            for i, atom_idx in enumerate(sorted(atoms)):
                label = f"p{i+1}"
                lines.append(f"{label}: POSITION ATOM={atom_idx + 1}")
        
        # Distances
        if 'distances' in self.selection_dict:
            sel_dict = self.selection_dict['distances']
            sel = sel_dict['selection']
            cutoff = sel_dict.get('cutoff', None)  # If no cutoff, perhaps all pairs, but assume required
            if cutoff is None:
                raise ValueError("Cutoff must be provided for distances.")
            periodic = sel_dict.get('periodic', True)
            include_bonded = sel_dict.get('bonded', True)  # But in example False
            
            atoms = self.top.select(sel)
            if len(atoms) == 0:
                raise ValueError(f"No atoms selected for distances with '{sel}'")
            
            # Compute pairwise distances to find pairs within cutoff
            pairs_list = list(itertools.combinations(atoms, 2))
            if len(pairs_list) > 0:
                dists = md.compute_distances(self.traj[0:1], pairs_list, periodic=periodic)[0]
                pairs = set()
                for ii in range(len(dists)):
                    if dists[ii] < cutoff:
                        a1, a2 = pairs_list[ii]
                        pairs.add((min(a1, a2), max(a1, a2)))
            else:
                pairs = set()
            
            # Exclude bonded if not include_bonded
            if not include_bonded:
                bond_set = {frozenset({b.atom1.index, b.atom2.index}) for b in self.top.bonds}
                pairs = {p for p in pairs if frozenset(p) not in bond_set}
            
            pbc_str = " NOPBC" if not periodic else ""
            for i, (a1, a2) in enumerate(sorted(pairs)):
                label = f"d{i+1}"
                lines.append(f"{label}: DISTANCE ATOMS={a1 + 1},{a2 + 1}{pbc_str}")
        
        # Angles
        if 'angles' in self.selection_dict:
            sel_dict = self.selection_dict['angles']
            sel = sel_dict['selection']
            bonded = sel_dict.get('bonded', True)
            if not bonded:
                raise NotImplementedError("Non-bonded angles not supported; too many combinations.")
            
            sel_atoms = set(self.top.select(sel))
            
            # Build adjacency list for selected atoms
            adj = defaultdict(list)
            for bond in self.top.bonds:
                a1, a2 = bond.atom1.index, bond.atom2.index
                if a1 in sel_atoms and a2 in sel_atoms:
                    adj[a1].append(a2)
                    adj[a2].append(a1)
            
            # Find triplets
            triplets = set()
            for j in sorted(sel_atoms):
                neigh = sorted(adj[j])
                for idx1 in range(len(neigh)):
                    for idx2 in range(idx1 + 1, len(neigh)):
                        i, k = neigh[idx1], neigh[idx2]
                        ordered = sorted([i, j, k])
                        triplets.add(tuple(ordered))
            
            for i, triplet in enumerate(sorted(triplets)):
                a1, a2, a3 = triplet
                label = f"a{i+1}"
                lines.append(f"{label}: ANGLE ATOMS={a1 + 1},{a2 + 1},{a3 + 1}")
        
        # Dihedrals
        if 'dihedrals' in self.selection_dict:
            sel_dict = self.selection_dict['dihedrals']
            sel = sel_dict['selection']
            bonded = sel_dict.get('bonded', True)
            if not bonded:
                raise NotImplementedError("Non-bonded dihedrals not supported; too many combinations.")
            
            # Use MDTraj's built-in for backbone dihedrals (phi, psi, omega)
            # Assumes selection is 'backbone' for protein; extend if needed
            phi_indices = md.compute_phi(self.traj)[0]
            psi_indices = md.compute_psi(self.traj)[0]
            omega_indices = md.compute_omega(self.traj)[0]
            
            all_indices = np.vstack((phi_indices, psi_indices, omega_indices))
            unique_dihedrals = set(tuple(row) for row in all_indices if all(idx in self.top.select(sel) for idx in row))
            
            for i, dihedral in enumerate(sorted(unique_dihedrals)):
                a1, a2, a3, a4 = dihedral
                label = f"t{i+1}"
                lines.append(f"{label}: TORSION ATOMS={a1 + 1},{a2 + 1},{a3 + 1},{a4 + 1}")
        
        # Write to file
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        
        print(f"Features file generated at {output_file}")
        