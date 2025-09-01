import mdtraj as md
import numpy as np
import torch
from biobb_pytorch.mdae.featurization.normalization import Normalization
from mlcolvar.core.transform.utils import Statistics

class Featurizer:
    """
    A class to extract geometric features (distances, angles, dihedrals) from MD trajectories using MDTraj.

    Physics behind the features:
      - Distance between atoms i and j: r_ij = |r_i - r_j| = sqrt((x_i-x_j)^2 + (y_i-y_j)^2 + (z_i-z_j)^2)
      - Angle between atoms i-j-k: θ = arccos((r_ji · r_jk) / (|r_ji| |r_jk|))
      - Dihedral for atoms i-j-k-l: φ = atan2(( (r_ji · (r_jk × r_kl)) * |r_jk| ), ((r_ji × r_jk) · (r_jk × r_kl)))

    Supports both tuple-based and dict-based feature definitions:
      - Tuple: e.g. ('name CA', 'name CB') or (0, 5)
      - Dict: e.g. {57: 'CA', 58: 'CA'} for distances
    """

    def __init__(self, trajectory_file, topology_file, input_labels_npy_path=None, input_weights_npy_path=None):
        """
        Initialize with an MDTraj Trajectory object.

        Parameters:
        -------------
        trajectory_file : str
            Path to the trajectory file (e.g., .dcd, .xtc).
        topology_file : str
            Path to the topology file (e.g., .pdb, .gro).
        """

        # Load trajectory and topology
        trajectory = md.load(trajectory_file, 
                             top=topology_file)

        self.trajectory = trajectory
        self.topology = trajectory.topology
        self.input_labels_npy_path = input_labels_npy_path
        self.input_weights_npy_path = input_weights_npy_path

        self.complete_top = md.Trajectory(xyz=trajectory.xyz[0], topology=trajectory.topology)


    def select_atoms(self, selection):
        """
        Convert a selection specifier into atom indices.

        Parameters:
        -------------
        selection : str or list[int] or np.ndarray
            If str: MDTraj topology query (e.g., 'name CA', 'resid 0 to 10').
            If list or array: explicit atom indices.

        Returns:
        --------
        np.ndarray
            Array of selected atom indices.
        """
        if isinstance(selection, str):
            idx = self.topology.select(selection)
        else:
            idx = np.array(selection, dtype=int)
        return idx
    
    def filter_topology(self, selection, topology):
        """
        Filter the topology based on a selection.

        Parameters:
        -------------
        selection : str
            MDTraj topology query (e.g., 'name CA', 'resid 0 to 10').
        topology : md.Topology
            MDTraj topology object.

        Returns:
        --------
        md.Topology
            Filtered topology.
        """
        idx = self.select_atoms(selection)
        return topology.atom_slice(idx)

    def _dicts_to_tuples(self, dict_list, expected_length):
        """
        Internal helper: convert list of dicts mapping resid->atom to tuple of atom indices.

        Each dict must have exactly expected_length entries. Keys are residue indices,
        values are atom names. Residue indices must match MDTraj topology resid numbering.
        """
        tuples = []
        for d in dict_list:
            if len(d) != expected_length:
                raise ValueError(f"Expected dict with {expected_length} entries, got {len(d)}")
            items = list(d.items())
            idxs = []
            for resid, atom_name in items:
                sel = f"resid {resid} and name {atom_name}"
                arr = self.select_atoms(sel)
                if len(arr) == 0:
                    raise ValueError(f"No atom found for {sel}")
                # take first match
                idxs.append(int(arr[0]))
            tuples.append(tuple(idxs))
        return tuples
    
    def idx_distances(self, pairs):
        """
        Convert pairs of atom indices to MDTraj topology indices.

        Parameters:
        -------------
        pairs : list of tuples or dicts
            Each tuple or dict must have 2 entries (resid, atom_name).

        Returns:
        --------
        np.ndarray
            Array of shape (n_pairs, 2) with atom indices.
        """
        if len(pairs) > 0 and isinstance(pairs[0], dict):
            pairs = self._dicts_to_tuples(pairs, 2)
        
        idx_pairs = []
        for a, b in pairs:
            ia = int(self.select_atoms(a)) if isinstance(a, str) else int(a)
            ib = int(self.select_atoms(b)) if isinstance(b, str) else int(b)
            idx_pairs.append((ia, ib))
        idx_pairs = np.array(idx_pairs)
        return idx_pairs

    def compute_distances(self, idx_pairs, cutoff, periodic: bool = True):
        """
        Compute inter-atomic distances for given pairs.

        Accepts pairs as list of 2-tuples or list of dicts with 2 entries {resid: atom_name}.
        """
        distances = md.compute_distances(self.trajectory, idx_pairs, periodic=periodic)
        
        # apply cutoff and get only pairs within cutoff
        if cutoff is not None:
            keep_cols = np.any(distances < cutoff, axis=0) 
            idx_pairs = idx_pairs[keep_cols]
            distances = distances[:, keep_cols]

        return distances, idx_pairs

    def polar2cartesian(self, a):
        """
        Convert polar coordinates to Cartesian coordinates.

        Parameters:
        -------------
        a : np.ndarray
            Array of shape (n_frames, n_angles) representing polar coordinates.
            Each row corresponds to a frame, each column to an angle.
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_frames, n_angles * 2) representing Cartesian coordinates.
            Each row corresponds to a frame, each column to sin and cos a.
        """
        # Convert angles to radians
        a = np.deg2rad(a)
        # Compute sin and cos
        x = np.sin(a)
        y = np.cos(a)
        # Stack sin and cos values
        cart_angles = np.column_stack((x, y))
        return cart_angles

    def cartesian2polar(self, cart_angles):
        """
        Convert Cartesian coordinates to polar coordinates.

        Parameters:
        -------------
        cart_angles : np.ndarray
            Array of shape (n_frames, n_angles * 2) representing Cartesian coordinates.
            Each row corresponds to a frame, each column to sin and cos a.
        
        Returns:
        --------
        np.ndarray
            Array of shape (n_frames, n_angles) representing polar coordinates.
            Each row corresponds to a frame, each column to an angle.
        """
        # Compute angles from sin and cos
        angles = np.arctan2(cart_angles[:, 0], cart_angles[:, 1])
        # Convert angles to degrees
        angles = np.rad2deg(angles)
        return angles
    
    def idx_angles(self, triplets):
        """
        Convert triplets of atom indices to MDTraj topology indices.

        Parameters:
        -------------
        triplets : list of tuples or dicts
            Each tuple or dict must have 3 entries (resid, atom_name).

        Returns:
        --------
        np.ndarray
            Array of shape (n_triplets, 3) with atom indices.
        """
        if len(triplets) > 0 and isinstance(triplets[0], dict):
            triplets = self._dicts_to_tuples(triplets, 3)
        
        idx_triplets = []
        for i, j, k in triplets:
            ii = int(self.select_atoms(i)) if isinstance(i, str) else int(i)
            jj = int(self.select_atoms(j)) if isinstance(j, str) else int(j)
            kk = int(self.select_atoms(k)) if isinstance(k, str) else int(k)
            idx_triplets.append((ii, jj, kk))
        idx_triplets = np.array(idx_triplets)
        return idx_triplets

    def compute_angles(self, idx_triplets, periodic: bool = True):
        """
        Compute angles between triplets of atoms.

        Accepts triplets as list of 3-tuples or list of dicts with 3 entries {resid: atom_name}.
        """
        
        return md.compute_angles(self.trajectory, idx_triplets, periodic=periodic)

    def idx_dihedrals(self, quadruplets):
        """
        Convert quadruplets of atom indices to MDTraj topology indices.

        Parameters:
        -------------
        quadruplets : list of tuples or dicts
            Each tuple or dict must have 4 entries (resid, atom_name).

        Returns:
        --------
        np.ndarray
            Array of shape (n_quadruplets, 4) with atom indices.
        """
        if len(quadruplets) > 0 and isinstance(quadruplets[0], dict):
            quadruplets = self._dicts_to_tuples(quadruplets, 4)
        
        idx_quads = []
        for i, j, k, l in quadruplets:
            ii = int(self.select_atoms(i)) if isinstance(i, str) else int(i)
            jj = int(self.select_atoms(j)) if isinstance(j, str) else int(j)
            kk = int(self.select_atoms(k)) if isinstance(k, str) else int(k)
            ll = int(self.select_atoms(l)) if isinstance(l, str) else int(l)
            idx_quads.append((ii, jj, kk, ll))
        idx_quads = np.array(idx_quads)
        return idx_quads

    def compute_dihedrals(self, idx_quads, periodic: bool = True):
        """
        Compute dihedral angles for quadruplets of atoms.

        Accepts quads as list of 4-tuples or list of dicts with 4 entries {resid: atom_name}.
        """
        
        return md.compute_dihedrals(self.trajectory, idx_quads, periodic=periodic)

    def compute_cartesian(self, indices):
        """
        Compute Cartesian coordinates for selected atoms.

        Parameters:
        -------------
        indices : list[int]
            List of atom indices to compute Cartesian coordinates for.

        Returns:
        --------
        np.ndarray
            Cartesian coordinates of the selected atoms.
        """
        return self.trajectory.xyz[:, indices, :]
    
    def combine_features(self, *feature_arrays):
        """
        Concatenate multiple feature arrays along the feature axis.
        """
        return np.concatenate(feature_arrays, axis=1)
    
    def timelag(self, data: np.ndarray, lag: int):
        """
        Split into X and Y where Y[t] = X[t+lag].

        Parameters
        ----------
        data : np.ndarray, shape (n_times, n_features)
        lag : int

        Returns
        -------
        X : np.ndarray, shape (n_times-lag, n_features)
        Y : np.ndarray, shape (n_times-lag, n_features)
        """
        if lag < 1 or lag >= data.shape[0]:
            raise ValueError("lag must be between 1 and n_times-1")

        X = data[:-lag]
        Y = data[lag:]
        return X, Y
    
    def get_n_features(self):
        """
        Get the number of features in the combined feature array.
        """
        return self.n_features
    
    def get_n_frames(self):
        """
        Get the number of frames in the combined feature array.
        """
        return self.n_frames
    
    def get_atom_info(self, selection):
        """
        Get the atom information from the topology.
        """
        idx = self.select_atoms(selection)
        top = self.trajectory.atom_slice(idx)
        atom_info = []
        for i in top.topology.atoms:
            atom_info.append([i.name, i.residue.name, i.residue.index+1])
        return np.array(atom_info, dtype=object)
    
    def set_statistics(self, combined: np.ndarray, feature_dict: dict):
        """
        Set statistics for the combined feature array.
        """
        stats = Statistics(torch.FloatTensor(combined)).to_dict()
        stats['shape'] = [self.n_frames, self.n_features]
        stats['selection'] = feature_dict['cartesian']['selection'] if 'cartesian' in feature_dict else None
        stats['topology'] = self.filter_topology(stats.get('selection', "name CA"), self.complete_top)
        stats["parametric"] = [torch.mean(torch.from_numpy(combined.flatten())), torch.std(torch.from_numpy(combined.flatten()))]

        if self.idx_cartesian is not None:
            stats['cartesian_indices'] = self.idx_cartesian
        if self.idx_dist is not None:
            stats['distance_indices'] = self.idx_dist.tolist()
        if self.idx_triplets is not None:
            stats['angle_indices'] = self.idx_triplets.tolist()
        if self.idx_quads is not None:
            stats['dihedral_indices'] = self.idx_quads.tolist()
        
        return stats
  
    def compute_features(self, feature_dict: dict):
        """
        Compute and combine multiple feature types in one call.

        feature_dict keys:
          - 'distances': list of tuple or dict (2 entries)
          - 'angles':    list of tuple or dict (3 entries)
          - 'dihedrals': list of tuple or dict (4 entries)

        Returns:
          combined : np.ndarray shape=(n_frames, total_features)
          features  : dict mapping feature type to its array
        """
        self.idx_cartesian = None
        self.idx_dist = None
        self.idx_triplets = None
        self.idx_quads = None

        self.features = {}
        arrays = []

        if 'distances' in feature_dict:
            self.idx_dist = self.idx_distances(feature_dict['distances']['pairs'])
            d, self.idx_dist = self.compute_distances(self.idx_dist, 
                                                 cutoff=feature_dict['distances']['cutoff'], 
                                                 periodic=feature_dict['distances']['periodic'])
            self.n_distances = d.shape[1]
            self.features['distances'] = d
            arrays.append(d)

        if 'angles' in feature_dict:
            self.idx_triplets = self.idx_angles(feature_dict['angles']['triplets'])
            a = self.compute_angles(self.idx_triplets, 
                                    periodic=feature_dict['angles']['periodic'])
            self.n_angles = a.shape[1]
            a = self.polar2cartesian(a)
            self.features['angles'] = a
            arrays.append(a)

        if 'dihedrals' in feature_dict:
            self.idx_quads = self.idx_dihedrals(feature_dict['dihedrals']['quadruplets'])
            phi = self.compute_dihedrals(self.idx_quads, 
                                         periodic=feature_dict['dihedrals']['periodic'])
            self.n_dihedrals = phi.shape[1]
            phi = self.polar2cartesian(phi)
            self.features['dihedrals'] = phi
            arrays.append(phi)

        if 'cartesian' in feature_dict:
            self.idx_cartesian = feature_dict['cartesian']['indices']
            cart = self.compute_cartesian(self.idx_cartesian)
            self.features['cartesian'] = cart.reshape(self.trajectory.n_frames, -1)
            self.n_cartesian = cart.shape[1]
            arrays.append(self.features['cartesian'])

        combined = self.combine_features(*arrays)

        self.n_features = combined.shape[1]
        self.n_frames = combined.shape[0]

        stats = self.set_statistics(combined, feature_dict)

        if self.input_labels_npy_path:
            labels = np.load(self.input_labels_npy_path)

        if self.input_weights_npy_path:
            weights = np.load(self.input_weights_npy_path)
        
        if 'norm_in' in feature_dict.get('options', {}):

            if feature_dict['options']['norm_in']['mode'] != 'custom':
                
                feature_dict['options']['norm_in']['stats'] = stats

            norm_in = Normalization(combined.shape[1], **feature_dict['options']['norm_in'])

            combined = norm_in(torch.FloatTensor(combined))
            combined = combined.numpy()

        # Add timelag features if specified
        if 'timelag' in feature_dict.get('options', {}):
            
            lag = feature_dict['options']['timelag']
            combined, combined_lag = self.timelag(combined, lag)

            dataset = {"data" : combined, "target": combined_lag}

            if self.input_labels_npy_path:
                labels, labels_lag = self.timelag(labels, lag)
                dataset["labels"] = labels_lag

            if self.input_weights_npy_path:
                weights, weights_lag = self.timelag(weights, lag)
                dataset["weights"] = weights_lag

            return dataset, stats

        else:
            dataset = {"data": combined}
            
            if self.input_labels_npy_path:
                dataset["labels"] = labels
            if self.input_weights_npy_path:
                dataset["weights"] = weights

            return dataset, stats


# Usage example
# traj = md.load("/home/pzanders/Documents/Simulations/GodMD/domini/1NE4_6NO7_b.dcd",
#                 top="/home/pzanders/Documents/Simulations/GodMD/domini/1NE4_6NO7_b.godmd.pdb")

# featurizer = MDFeaturizer(traj)

# atom_pairs = [(57, 78), (57, 79)]
# atom_triplets = [(57, 78, 79), (57, 78, 80)]
# atom_quadruplets = [(57, 78, 79, 80), (57, 78, 79, 81)]

# feature_dict = {
#     'distances': {"pairs": atom_pairs, "cutoff": 0.5, "periodic": True},
#     'angles':    {"triplets": atom_triplets, "periodic": True},
#     'dihedrals': {"quadruplets": atom_quadruplets, "periodic": True}
# }
# combined, details = featurizer.compute_features(feature_dict)
