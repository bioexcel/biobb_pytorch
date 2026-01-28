from biobb_common.generic.biobb_object import BiobbObject
from biobb_common.tools.file_utils import launchlogger
import torch
import numpy as np
import mdtraj as md
import os


class Feat2Traj(BiobbObject):
    """
    | biobb_pytorch Feat2Traj
    | Converts a .pt file (features) to a trajectory using cartesian indices and topology from the stats file.
    | Converts a .pt file (features) to a trajectory using cartesian indices and topology from the stats file.

    Args:
        input_results_npz_path (str): Path to the input reconstructed results file (.npz), typically containing an 'xhat' array. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_results.npz>`_. Accepted formats: npz (edam:format_2333).
        input_stats_pt_path (str): Path to the input model statistics file (.pt) containing cartesian indices and optionally topology. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_model.pt>`_. Accepted formats: pt (edam:format_2333).
        input_topology_path (str) (optional): Path to the topology file (PDB) used if no suitable topology is found in the stats file. Used if no topology is found in stats. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/mdae/ref_input_topology.pdb>`_. Accepted formats: pdb (edam:format_1476).
        output_traj_path (str): Path to save the trajectory in xtc/pdb/dcd format. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.xtc>`_. Accepted formats: xtc (edam:format_3875), pdb (edam:format_1476), dcd (edam:format_3878).
        output_top_path (str) (optional): Path to save the output topology file (pdb). Used if trajectory format requires separate topology. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/mdae/output_model.pdb>`_. Accepted formats: pdb (edam:format_1476).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **restart** (*bool*) - (False) [WF property] Do not execute if output files exist.

    Examples:
        This example shows how to use the Feat2Traj class to convert a .pt file (features) to a trajectory using cartesian indices and topology from the stats file::

            from biobb_pytorch.mdae.feat2traj import feat2traj

            input_results_npz_path='input_results.npz'
            input_stats_pt_path='input_model.pt'
            input_topology_path='input_topology.pdb'
            output_traj_path='output_model.xtc'
            output_top_path='output_model.pdb'

            prop={}

            feat2traj(input_results_npz_path=input_results_npz_path,
                    input_stats_pt_path=input_stats_pt_path,
                    input_topology_path=input_topology_path,
                    output_traj_path=output_traj_path,
                    output_top_path=output_top_path,
                    properties=prop)

    Info:
        * wrapped_software:
            * name: PyTorch
            * version: >=1.6.0
            * license: BSD 3-Clause
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl
    """

    def __init__(
        self,
        input_results_npz_path: str,
        input_stats_pt_path: str,
        input_topology_path: str = None,
        output_traj_path: str = None,
        output_top_path: str = None,
        properties: dict = None,
        **kwargs,
    ) -> None:
        properties = properties or {}
        super().__init__(properties)

        self.input_results_npz_path = input_results_npz_path
        self.input_stats_pt_path = input_stats_pt_path
        self.input_topology_path = input_topology_path
        self.output_traj_path = output_traj_path
        self.output_top_path = output_top_path
        self.properties = properties.copy()
        self.locals_var_dict = locals().copy()
        self.io_dict = {
            "in": {
                "input_results_npz_path": input_results_npz_path,
                "input_stats_pt_path": input_stats_pt_path,
                "input_topology_path": input_topology_path,
            },
            "out": {
                "output_traj_path": output_traj_path,
                "output_top_path": output_top_path,
            },
        }
        self.check_properties(properties)
        self.check_arguments()

    @launchlogger
    def launch(self) -> int:
        """
        Execute the :class:`Feat2Traj` class and its `.launch()` method.
        """
        # Load features
        features = np.load(self.input_results_npz_path)
        features = features['xhat']

        # Load stats and extract cartesian indices and topology
        stats = torch.load(self.input_stats_pt_path,
                           weights_only=False)
        cartesian_indices = None
        topology = None
        if isinstance(stats, dict):
            if 'cartesian_indices' in stats:
                cartesian_indices = stats['cartesian_indices']
                topology = stats['topology']

        else:
            raise ValueError('No cartesian indices found in stats file.')
        cartesian_indices = np.array(cartesian_indices)

        n_atoms = len(cartesian_indices)
        n_frames = features.shape[0]
        coords = features.reshape((n_frames, n_atoms, 3))

        # Try to use topology from stats file if present
        top = None
        if topology is not None:
            try:
                # If topology is a serialized MDTraj Topology, try to load it
                if isinstance(topology, md.Trajectory):
                    top = topology.topology
                elif isinstance(topology, str) and os.path.exists(topology):
                    top = md.load_topology(topology)
                elif isinstance(topology, dict) and 'pdb_string' in topology:
                    import io
                    top = md.load(io.StringIO(topology['pdb_string']), format='pdb').topology
            except Exception as e:
                print(f"Warning: Could not load topology from stats file: {e}")
                top = None

        # If not found, try input_topology_path
        if top is None and self.input_topology_path is not None and os.path.exists(self.input_topology_path):
            top = md.load_topology(self.input_topology_path)
        # Fallback: create a fake topology
        if top is None:
            top = md.Topology()
            chain = top.add_chain()
            res = top.add_residue('RES', chain)
            for i in range(n_atoms):
                top.add_atom('CA', element=md.element.carbon, residue=res)
        traj = md.Trajectory(xyz=coords, topology=top)

        if self.output_traj_path:
            ext = os.path.splitext(self.output_traj_path)[1]
            if ext == '.xtc':
                traj.save_xtc(self.output_traj_path)
                traj[0].save_pdb(self.output_top_path)
            elif ext == '.dcd':
                traj.save_dcd(self.output_traj_path)
                traj[0].save_pdb(self.output_top_path)
            elif ext == '.pdb':
                traj.save_pdb(self.output_traj_path)
            else:
                raise ValueError(f'Unknown trajectory extension: {ext}')
        return 0


def feat2traj(
    input_results_npz_path: str,
    input_stats_pt_path: str,
    input_topology_path: str = None,
    output_traj_path: str = None,
    output_top_path: str = None,
    properties: dict = None,
    **kwargs,
) -> int:
    """Create the :class:`Feat2Traj <Feat2Traj.Feat2Traj>` class and
    execute the :meth:`launch() <Feat2Traj.feat2traj.launch>` method."""
    return Feat2Traj(**dict(locals())).launch()


feat2traj.__doc__ = Feat2Traj.__doc__
main = Feat2Traj.get_main(feat2traj, "Converts a .pt file (features) to a trajectory using cartesian indices and topology from the stats file.")

if __name__ == "__main__":
    main()
