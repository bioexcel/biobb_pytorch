import mdtraj as md
import itertools

class MDTopologySelector:
    """
    A class to load an MDTraj topology and extract atom pairs (bonds or distances), triplets (angles or arbitrary triples),
    and quads (torsions or arbitrary quadruplets) for a given atom selection.
    """
    def __init__(self, topology):
        """
        Parameters
        ----------
        topology : str | md.Trajectory | md.Topology
            Path to a structure file (e.g., .pdb, .gro), an MDTraj Trajectory, or an MDTraj Topology.
        """
        
        if isinstance(topology, md.Trajectory):
            self.topology = topology.topology
        elif isinstance(topology, md.Topology):
            self.topology = topology
        elif isinstance(topology, str):
            traj = md.load(topology)
            self.topology = traj.topology
        else:
            raise ValueError("`topology` must be a file path, md.Trajectory, or md.Topology instance.")

        # Precompute bond list as tuples of atom indices
        self.bonds = [(b.atom1.index, b.atom2.index) for b in self.topology.bonds]

    def select(self, selection):
        """
        Select atom indices matching an MDTraj selection string.

        Parameters
        ----------
        selection : str
            MDTraj selection syntax, e.g., "backbone", "name CA", etc.

        Returns
        -------
        numpy.ndarray of int
            Array of atom indices.
        """
        return self.topology.select(selection)

    def get_atom_pairs(self, selection, bonded=True):
        """
        Get atom pairs for a selection.

        Parameters
        ----------
        selection : str
            MDTraj selection syntax.
        bonded : bool, default=True
            If True, return only bonded pairs. If False, return all unique pairs (nonbonded).

        Returns
        -------
        List of tuple(int, int)
            Each tuple is (i, j) of atom indices.
        """
        sel = list(self.select(selection))
        if bonded:
            sel_set = set(sel)
            atom_pairs = [(i, j) for (i, j) in self.bonds if i in sel_set and j in sel_set]
        else:
            atom_pairs = list(itertools.combinations(sel, 2))
        
        self.n_distances = len(atom_pairs)
        
        return atom_pairs

    def get_triplets(self, selection, bonded=True):
        """
        Get atom triplets for a selection.

        Parameters
        ----------
        selection : str
            MDTraj selection syntax.
        bonded : bool, default=True
            If True, return triplets that form angles (i-j-k where i-j and j-k are bonds).
            If False, return all unique triplets.

        Returns
        -------
        List of tuple(int, int, int)
            Each tuple is (i, j, k).
        """
        sel = list(self.select(selection))
        if not bonded:
            return list(itertools.combinations(sel, 3))

        sel_set = set(sel)
        # build adjacency dict
        nbrs = {a: set() for a in sel_set}
        for i, j in self.bonds:
            if i in sel_set and j in sel_set:
                nbrs[i].add(j)
                nbrs[j].add(i)

        triplets = set()
        for j in sel_set:
            for i in nbrs[j]:
                for k in nbrs[j]:
                    if i != k:
                        triplets.add((i, j, k))

        self.n_angles = len(list(triplets))

        return list(triplets)

    def get_quads(self, selection, bonded=True):
        """
        Get atom quads for a selection.

        Parameters
        ----------
        selection : str
            MDTraj selection syntax.
        bonded : bool, default=True
            If True, return quads that form torsions (i-j-k-l where i-j, j-k, k-l are bonds).
            If False, return all unique quadruplets.

        Returns
        -------
        List of tuple(int, int, int, int)
            Each tuple is (i, j, k, l).
        """
        sel = list(self.select(selection))
        if not bonded:
            return list(itertools.combinations(sel, 4))

        sel_set = set(sel)
        nbrs = {a: set() for a in sel_set}
        for i, j in self.bonds:
            if i in sel_set and j in sel_set:
                nbrs[i].add(j)
                nbrs[j].add(i)

        quads = set()
        for j in sel_set:
            for k in nbrs[j]:
                for i in nbrs[j]:
                    if i == k:
                        continue
                    for l in nbrs[k]:
                        if l == j:
                            continue
                        quads.add((i, j, k, l))

        self.n_dihedrals = len(quads)

        return list(quads)

    def get_atom_indices(self, selection):
        """
        Get atom indices for a selection.

        Parameters
        ----------
        selection : str
            MDTraj selection syntax.

        Returns
        -------
        List of int
            Atom indices.
        """

        atom_indices = list(self.select(selection))
        self.n_atoms = len(atom_indices)

        return atom_indices
    
    def topology_indexing(self, config):
        """
        Get the topology indexing for a given configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing selections for cartesian, distances, angles, and dihedrals.

        Returns
        -------
        Dict of indices
            Topology indices.
        """
        
        self.config = config

        self.topology_idx = {}
        if 'cartesian' in self.config:
            sel = self.config['cartesian']['selection']
            idx = self.get_atom_indices(sel)

            fit_sel = self.config['cartesian'].get('fit_selection', None)
            if fit_sel is not None:
                fit_idx = self.get_atom_indices(fit_sel)
                self.topology_idx['cartesian']['fit_selection'] = fit_idx
                self.topology_idx['cartesian'] = {'selection': sel, 'indices': idx, 'fit_selection': fit_idx}
            else:
                self.topology_idx['cartesian'] = {'selection': sel, 'indices': idx}

        if 'distances' in self.config:
            sel    = self.config['distances']['selection']
            bonded = self.config['distances'].get('bonded', False)
            pairs  = self.get_atom_pairs(sel, bonded=bonded)
            # pull other args
            cutoff  = self.config['distances'].get('cutoff', None)
            periodic = self.config['distances'].get('periodic', False)
            args = {'selection': sel, 
                    'pairs': pairs}
            if cutoff is not None:   args['cutoff']  = cutoff
            args['periodic'] = periodic
            self.topology_idx['distances'] = args

        if 'angles' in self.config:
            sel     = self.config['angles']['selection']
            bonded  = self.config['angles'].get('bonded', True)
            triplets = self.get_triplets(sel, bonded=bonded)
            periodic = self.config['angles'].get('periodic', False)
            self.topology_idx['angles'] = {
                'selection': sel,
                'triplets': triplets,
                'periodic': periodic
            }

        if 'dihedrals' in self.config:
            sel       = self.config['dihedrals']['selection']
            bonded    = self.config['dihedrals'].get('bonded', True)
            quads     = self.get_quads(sel, bonded=bonded)
            periodic  = self.config['dihedrals'].get('periodic', False)
            self.topology_idx['dihedrals'] = {
                'selection': sel,
                'quadruplets': quads,
                'periodic': periodic
            }

        # collect any options
        if 'options' in self.config:
            self.topology_idx['options'] = self.config['options']
        
        return self.topology_idx

