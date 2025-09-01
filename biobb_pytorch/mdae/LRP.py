import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from typing import Optional
from biobb_common.configuration import settings
from biobb_common.tools.file_utils import launchlogger
from biobb_common.tools import file_utils as fu
from biobb_pytorch.mdae.utils.log_utils import get_size
from biobb_common.generic.biobb_object import BiobbObject
from torch.utils.data import DataLoader
from mlcolvar.data import DictDataset
from biobb_pytorch.mdae.explainability.layerwise_relevance_prop import lrp_encoder  

class LRP(BiobbObject):
    """
    | biobb_pytorch LRP
    | Performs Layer-wise Relevance Propagation on a trained autoencoder encoder.
    | Performs Layer-wise Relevance Propagation on a trained autoencoder encoder.

    Args:
        input_model_pth_path (str): Path to the input model file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        input_dataset_pt_path (str): Path to the input dataset file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pt>`_. Accepted formats: pt (edam:format_2333).
        output_results_npz_path (str) (Optional): Path to the output results file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_results.npz>`_. Accepted formats: npz (edam:format_2333).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **Dataset** (*dict*) - ({}) Dataset options from mlcolvar, including batch_size.

    Examples:
        This example shows how to use the LRP class to perform Layer-wise Relevance Propagation::

            from biobb_pytorch.mdae.explainability import relevancePropagation
            
            input_model_pth_path='input_model.pth'
            input_dataset_pt_path='input_dataset.pt'
            output_results_npz_path='output_results.npz'

            prop={
                'Dataset': {
                    'batch_size': 32
                }
            }

            # For API usage, output can be None to avoid saving
            instance = LRP(input_model_pth_path=input_model_pth_path,
                           input_dataset_pt_path=input_dataset_pt_path,
                           output_results_npz_path=None,
                           properties=prop)
            instance.launch()
            results = instance.results  # Access the results dict

            # Or to save, provide output and call launch
            relevancePropagation(input_model_pth_path=input_model_pth_path,
                                 input_dataset_pt_path=input_dataset_pt_path,
                                 output_results_npz_path=output_results_npz_path,
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
        input_model_pth_path: str,
        input_dataset_pt_path: str,
        output_results_npz_path: Optional[str] = None,
        properties: dict = None,
    ) -> None:
        
        properties = properties or {}

        super().__init__(properties)

        self.input_model_pth_path = input_model_pth_path
        self.input_dataset_pt_path = input_dataset_pt_path
        self.output_results_npz_path = output_results_npz_path
        self.properties = properties.copy()
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = {
            "in": {
                "input_model_pth_path": input_model_pth_path,
                "input_dataset_pt_path": input_dataset_pt_path,
            },
            "out": {},
        }

        if output_results_npz_path:
            self.io_dict["out"]["output_results_npz_path"] = output_results_npz_path

        self.Dataset = self.properties.get('Dataset', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = None

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

    def load_model(self):
        return torch.load(self.io_dict["in"]["input_model_pth_path"])

    def mask_idx(self, dataset: dict, indices: np.ndarray) -> dict:
        """
        Mask the dataset (dict) for all keys.
        """
        for key in dataset.keys():
            dataset[key] = dataset[key][indices]
        return dataset

    def load_dataset(self):
        dataset = torch.load(self.io_dict["in"]["input_dataset_pt_path"])

        if self.Dataset.get('indices', None):
            if isinstance(self.Dataset['indices'], list):
                indices = np.array(self.Dataset['indices'])
            elif isinstance(self.Dataset['indices'], np.ndarray):
                indices = self.Dataset['indices']
            dataset = self.mask_idx(dataset, indices)

        return DictDataset(dataset)

    def create_dataloader(self, dataset):
        ds_cfg = self.properties['Dataset']
        return DataLoader(
            dataset,
            batch_size=ds_cfg.get('batch_size', 16),
            shuffle=False
        )

    def compute_global_importance(self, model, dataloader, latent_index=None):
        all_R0 = []
        for batch in dataloader:
            X_batch = batch['data'].to(self.device)  # Assuming DictDataset with 'data' key
            R0 = lrp_encoder(model, X_batch, latent_index=latent_index)
            all_R0.append(R0.cpu())  # Move to CPU to save GPU memory
        R0_all = torch.cat(all_R0, dim=0)  # [total_samples, in_dim]
        
        # Reshape assuming features grouped by 3 (e.g., coordinates); adjust if needed
        num_features = R0_all.size(1) // 3
        R0_all = R0_all.reshape(-1, num_features, 3)
        R0_mean = R0_all.mean(dim=2)  # [total_samples, num_features]
        
        global_importance = R0_mean.abs().mean(dim=0)  # [num_features]
        global_importance_raw = global_importance.detach().numpy()
        
        # Normalize
        min_val = global_importance_raw.min()
        max_val = global_importance_raw.max()
        global_range = max_val - min_val + 1e-10  # Avoid division by zero
        global_importance_norm = (global_importance_raw - min_val) / global_range
        
        return {
            "global_importance": global_importance_norm,
            "global_importance_raw": global_importance_raw,
        }

    @launchlogger
    def launch(self) -> int:
        """
        Execute the :class:`LRP` class and its `.launch()` method.
        """

        fu.log(f'## BioBB Layer-wise Relevance Propagation ##', self.out_log)

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        # load the model
        fu.log(f'Load model from {os.path.abspath(self.io_dict["in"]["input_model_pth_path"])}', self.out_log)
        model = self.load_model()

        # load the dataset
        fu.log(f'Load dataset from {os.path.abspath(self.io_dict["in"]["input_dataset_pt_path"])}', self.out_log)
        dataset = self.load_dataset()
        
        # create the dataloader
        fu.log(f'Start LRP analysis...', self.out_log)
        dataloader = self.create_dataloader(dataset)

        # Compute LRP
        self.results = self.compute_global_importance(model, dataloader, latent_index=None)
        
        # Save the results if path provided
        if self.output_results_npz_path:
            np.savez_compressed(self.io_dict["out"]["output_results_npz_path"], **self.results)
            fu.log(f'Results saved to {os.path.abspath(self.io_dict["out"]["output_results_npz_path"])}', self.out_log)
            fu.log(f'File size: {get_size(self.io_dict["out"]["output_results_npz_path"])}', self.out_log)

        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        output_created = bool(self.output_results_npz_path)
        self.check_arguments(output_files_created=output_created, raise_exception=False)

        return 0 
        
def relevancePropagation(
    properties: dict,
    input_model_pth_path: str,
    input_dataset_pt_path: str,
    output_results_npz_path: Optional[str] = None,
) -> int:
    """
    Execute the :class:`LRP <LRP>` class and
    execute the :meth:`launch() <LRP.launch>` method.
    """
    return LRP(
        input_model_pth_path=input_model_pth_path,
        input_dataset_pt_path=input_dataset_pt_path,
        output_results_npz_path=output_results_npz_path,
        properties=properties,
    ).launch()

relevancePropagation.__doc__ = LRP.__doc__

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

    required_args.add_argument(
        "-m",
        "--input_model_pth_path",
        required=True,
        help="Model file path"
    )

    required_args.add_argument(
        "-f",
        "--input_dataset_pt_path",
        required=True,
        help="Dataset file path"
    )

    required_args.add_argument(
        "-or",
        "--output_results_npz_path",
        required=False,
        help="Results file path"
    )

    parser.add_argument(
        "-p",
        "--properties",
        required=False,
        help="Additional properties for the LRP object.",
    )

    args = parser.parse_args()
    config = args.config if args.config else None
    properties = settings.ConfReader(config=config).get_prop_dic()

    relevancePropagation(
        input_model_pth_path=args.input_model_pth_path,
        input_dataset_pt_path=args.input_dataset_pt_path,
        output_results_npz_path=args.output_results_npz_path,
        properties=properties,
    )

if __name__ == "__main__":
    main()