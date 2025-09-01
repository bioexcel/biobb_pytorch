#!/usr/bin/env python3

import os
import torch
from biobb_pytorch.mdae.featurization.topology_selector import MDTopologySelector
from biobb_pytorch.mdae.featurization.featurizer import Featurizer
from biobb_common.configuration import settings
from biobb_common.generic.biobb_object import BiobbObject
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
import argparse
import numpy as np
from typing import Optional, Dict, Any
from biobb_pytorch.mdae.utils.log_utils import get_size

class MDFeaturePipeline(BiobbObject):
    """
    | biobb_pytorch MDFeaturePipeline
    | Obtain the Molecular Dynamics Features for PyTorch model training.
    | Obtain the Molecular Dynamics Features for PyTorch model training.

    Args:
        input_trajectory_path (str) (Optional): Path to the input train data file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/data/mdae/train_mdae_traj.xtc>`_. Accepted formats: xtc,dcd (edam:format_4003).
        input_topology_path (str): Path to the input model file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pdb>`_. Accepted formats: pdb (edam:format_2333).
        output_dataset_pt_path (str): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pt>`_. Accepted formats: pt (edam:format_2333).
        output_stats_pt_path (str): Path to the output model statistics file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pt>`_. Accepted formats: pt (edam:format_2333).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **cartesian** (*dict*) - ({'selection': str, 'fit_selection': str}) select the atoms for obtaining cartesian coordinates
            * **distances** (*dict*) - ({'selection': str, 'cutoff': float, 'periodic': bool}) select the atoms for obtaining distances
            * **angles** (*dict*) - ({'selection': str, 'periodic': bool}) select the atoms for obtaining angles
            * **dihedrals** (*dict*) - ({'selection': str, 'periodic': bool}) select the atoms for obtaining dihedrals
            * **options** (*dict*) - ({ 'timelag': int, 'norm_in': {'min_max'}}) options for dataset processing

    Examples:
        This is a use case of how to use the building block from Python::

            from biobb_pytorch.mdae.MDFeaturePipeline import MDFeaturizer

            prop = {
                'cartesian': {'selection': 'name CA'},
                'distances': {'selection': 'name CA', 
                              'cutoff': 0.4, 
                              'periodic': True,
                              'bonded': False},
                'angles': {'selection': 'backbone', 
                           'periodic': True, 
                           'bonded': True},
                'dihedrals': {'selection': 'backbone', 
                              'periodic': True, 
                              'bonded': True},
                'options': {'timelag': 10, 
                            'norm_in': {'mode': 'min_max'}
                           }
            }
            
            MDFeaturizer(input_trajectory_path=trajectory_file,
                         input_topology_path=topology_file,
                         output_dataset_pt_path=output_file,
                         output_stats_pt_path=output_stats_file,
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
    def __init__(self,
                 input_topology_path:   str,
                 output_dataset_pt_path:    str,
                 output_stats_pt_path: str,
                 properties: dict,
                 input_trajectory_path: Optional[str] = None,
                 input_labels_npy_path: Optional[str] = None,
                 input_weights_npy_path: Optional[str] = None
    ) -> None:

        properties = properties or {}
                
        super().__init__(properties)
        
        self.input_trajectory_path   = input_trajectory_path if input_trajectory_path else input_topology_path
        self.input_topology_path     = input_topology_path
        self.input_labels_npy_path   = input_labels_npy_path
        self.input_weights_npy_path  = input_weights_npy_path
        self.output_dataset_pt_path  = output_dataset_pt_path
        self.output_stats_pt_path    = output_stats_pt_path
        self.config                  = properties.copy()
        self.locals_var_dict         = locals().copy()

        # Input/Output files
        self.io_dict = {
            "in": {
                "input_trajectory_path": input_trajectory_path,
                "input_topology_path": input_topology_path,
                "input_labels_npy_path": input_labels_npy_path,
                "input_weights_npy_path": input_weights_npy_path,
            },
            "out": {
                "output_dataset_pt_path": output_dataset_pt_path,
                "output_stats_pt_path": output_stats_pt_path,
            },
        }

        # build the per-feature arguments
        self.feature_types = ["cartesian", "distances", "angles", "dihedrals"]
        self.cartesian: dict = properties.get("cartesian", {})
        self.distances: dict = properties.get("distances", {})
        self.angles:    dict = properties.get("angles", {})
        self.dihedrals: dict = properties.get("dihedrals", {})
        self.options:   dict = properties.get("options", {})

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

        # Topology indices  
        self.topology_indices()

        # Featurizer
        self.featurize_trajectory()

    @launchlogger
    def topology_indices(self) -> Dict[str, Any]:

        fu.log("## BioBB Featurization - MDFeaturePipeline ##", self.out_log)

        fu.log(f"Obtaining the topology information from {self.input_topology_path}", self.out_log)

        self.topology = MDTopologySelector(self.input_topology_path)
        self.features_idx_dict = self.topology.topology_indexing(self.config)

        fu.log("Available Topology Properties:", self.out_log)
        fu.log(f"  - Number of chains: {self.topology.topology.n_chains}", self.out_log)
        fu.log(f"  - Number of residues: {self.topology.topology.n_residues}", self.out_log)
        fu.log(f"  - Number of atoms: {self.topology.n_atoms}", self.out_log)
        try:
            fu.log(f"  - Number of distances: {self.topology.n_distances}", self.out_log)
        except AttributeError:
            fu.log("  - Number of distances: N/A", self.out_log)
        try:
            fu.log(f"  - Number of angles: {self.topology.n_angles}", self.out_log)
        except AttributeError:
            fu.log("  - Number of angles: N/A", self.out_log)
        try:
            fu.log(f"  - Number of dihedrals: {self.topology.n_dihedrals}", self.out_log)
        except AttributeError:
            fu.log("  - Number of dihedrals: N/A", self.out_log)

    @launchlogger
    def featurize_trajectory(self) -> None:

        self.featurizer = Featurizer(self.input_trajectory_path,
                                     self.input_topology_path,
                                     self.input_labels_npy_path,
                                     self.input_weights_npy_path,
                                     )
        
        fu.log(f"Available Trajectory Properties:", self.out_log)
        fu.log(f"   - Number of frames: {self.featurizer.trajectory.n_frames}", self.out_log)
        
        fu.log(f"Featurizing the trajectory {self.input_trajectory_path}", self.out_log)

        self.dataset, self.stats = self.featurizer.compute_features(self.features_idx_dict)

        if self.input_labels_npy_path:
            fu.log(f"Loading labels from {self.input_labels_npy_path}", self.out_log)
            self.dataset['labels'] =  np.load(self.input_labels_npy_path)

        if self.input_weights_npy_path:
            fu.log(f"Loading weights from {self.input_weights_npy_path}", self.out_log)
            self.dataset['weights'] = np.load(self.input_weights_npy_path)

        fu.log("Features:", self.out_log)
        for feature_type in self.feature_types:
            try:
                selection = getattr(self, feature_type).get("selection")
                shape = self.featurizer.features.get(feature_type, np.zeros((0, 0))).shape[1]
                fu.log(f"  {feature_type.capitalize()}:", self.out_log)
                fu.log(f"   - Topology Selection: {selection}", self.out_log)
                fu.log(f"   - Number of features: {shape}", self.out_log)
            except AttributeError:
                pass

        fu.log("Postprocessing:", self.out_log)
        fu.log(f"   - Normalization: {self.options.get("norm_in", {}).get("mode")}", self.out_log)
        fu.log(f"   - Timelag: {self.options.get("timelag", {})}", self.out_log)
        fu.log(f"Dataset Properties:", self.out_log)
        fu.log(f"   - Dataset: {self.dataset.keys()}", self.out_log)
        fu.log(f"   - Number of frames: {self.dataset["data"].shape[0]}", self.out_log)
        fu.log(f"   - Number of features: {self.dataset["data"].shape[1]}", self.out_log)
        

    @launchlogger
    def launch(self) -> int:
        """
        Execute the :class:`MDFeaturePipeline <MDFeaturePipeline.MDFeaturePipeline>` object
        """

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        torch.save(self.dataset, 
                   self.output_dataset_pt_path)
        
        fu.log(f"Dataset saved in .pt format in {os.path.abspath(self.io_dict["out"]["output_dataset_pt_path"])}", 
               self.out_log,
               )
        fu.log(f'File size: {get_size(self.io_dict["out"]["output_dataset_pt_path"])}', 
               self.out_log,
               )
        
        torch.save(self.stats,
                   os.path.splitext(self.output_stats_pt_path)[0] + ".pt")
        
        fu.log(f"Dataset statistics saved in .pt format in {os.path.abspath(self.io_dict["out"]["output_stats_pt_path"])}",
               self.out_log,
               )
        fu.log(f'File size: {get_size(self.io_dict["out"]["output_stats_pt_path"])}',
                self.out_log,
                )
        
        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)

        return 0 
        
def MDFeaturizer(
    input_topology_path: str,
    output_dataset_pt_path: str,
    output_stats_pt_path: str,
    properties: dict,
    input_trajectory_path: Optional[str] = None,
    input_labels_npy_path: Optional[str] = None,
    input_weights_npy_path: Optional[str] = None
) -> int:
    """
    Execute the :class:`MDFeaturePipeline <MDFeaturePipeline.MDFeaturePipeline>` class and
    execute the :meth:`launch() <MDFeaturePipeline.MDFeaturizer.launch>` method.
    """
    input_trajectory_path = input_trajectory_path if input_trajectory_path else input_topology_path
    return MDFeaturePipeline(
            input_trajectory_path=input_trajectory_path,
            input_topology_path=input_topology_path,  
            input_labels_npy_path=input_labels_npy_path,
            input_weights_npy_path=input_weights_npy_path,
            output_dataset_pt_path=output_dataset_pt_path,
            output_stats_pt_path=output_stats_pt_path,
            properties=properties,
    ).launch()

    MDFeaturizer.__doc__ = MDFeaturePipeline.__doc__

def main():
    """Command line execution of this building block. Please check the command line documentation."""

    parser = argparse.ArgumentParser(
        description="",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999),
    )

    parser.add_argument(
    "-c",
    "--config",
    required=False,
    help="This file can be a YAML file, JSON file or JSON string",
    )

    required_args = parser.add_argument_group("required arguments")
    parser.add_argument(
        "-f",
        "--input_trajectory_path",
        required=False,
        help="Trajectory file path"
    )

    required_args.add_argument(
        "-s",
        "--input_topology_path",
        required=True,
        help="topology file path"
    )

    required_args.add_argument(
        "-o",
        "--output_dataset_pt_path",
        required=True,
        help="Output pt file path"
    )

    parser.add_argument(
        "-p",
        "--properties",
        required=False,
        help="Additional properties for the MDFeaturizer object.",
    )

    parser.add_argument(
        "-l",
        "--input_labels_npy_path",
        required=True,
        help="Input labels .npy file path"
    )

    parser.add_argument(
        "-w",
        "--input_weights_npy_path",
        required=False,
        help="Input weights .npy file path"
    )

    args = parser.parse_args()
    config = args.config if args.config else None
    properties = settings.ConfReader(config=config).get_prop_dic()

    MDFeaturizer(
                input_trajectory_path=args.input_trajectory_path,
                input_labels_npy_path=args.input_labels_npy_path,
                input_weights_npy_path=args.input_weights_npy_path,
                input_topology_path=args.input_topology_path,  
                output_dataset_pt_path=args.output_dataset_pt_path,  
                properties=properties,
        )

if __name__ == "__main__":
    main()