import torch
from torch.utils.data import DataLoader
import importlib
import torch
from typing import Dict, Any, Type
import os
import argparse
from typing import Optional
from biobb_common.configuration import settings
from biobb_common.tools.file_utils import launchlogger
from biobb_common.tools import file_utils as fu
from biobb_pytorch.mdae.utils.log_utils import get_size
from biobb_common.generic.biobb_object import BiobbObject
import lightning.pytorch.callbacks as _cbs
import lightning.pytorch.loggers as _loggers
import lightning.pytorch.profilers as _profiler
from mlcolvar.utils.trainer import MetricsCallback
import lightning
from mlcolvar.data import DictModule
from mlcolvar.data import DictDataset
import numpy as np

class EvaluateEncoder(BiobbObject):
    """
    | biobb_pytorch EvaluateEncoder
    | Evaluates a PyTorch autoencoder from the given properties.
    | Evaluates a PyTorch autoencoder from the given properties.

    Args:
        input_model_pth_path (str): Path to the input model file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        input_dataset_pt_path (str): Path to the input dataset file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pt>`_. Accepted formats: pt (edam:format_2333).
        output_results_npz_path (str): Path to the output results file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_results.npz>`_. Accepted formats: npz (edam:format_2333).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **Dataset** (*dict*) - ({}) Dataset options from mlcolvar.

    Examples:
        This example shows how to use the EvaluateEncoder class to evaluate a PyTorch autoencoder model::

            from biobb_pytorch.mdae.evaluate_model import evaluateEncoder
            
            input_model_pth_path='input_model.pth'
            input_dataset_pt_path='input_dataset.npy'
            output_results_npz_path='output_results.npz'

            prop={
                'Dataset': {
                    'batch_size': 32
                }
            }
            
            evaluateEncoder(input_model_pth_path=input_model.pth,
                    input_dataset_pt_path=input_dataset.npy,
                    output_results_npz_path=output_results.npz,
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
        output_results_npz_path: str,
        properties: dict,
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
            "out": {
                "output_results_npz_path": output_results_npz_path,
            },
        }

        self.Dataset = self.properties.get('Dataset', {})
        self.results = None

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

    def load_model(self):
        return torch.load(self.io_dict["in"]["input_model_pth_path"])

    def load_dataset(self):
        dataset = torch.load(self.io_dict["in"]["input_dataset_pt_path"])
        return DictDataset(dataset)

    def create_dataloader(self, dataset):
        ds_cfg = self.properties['Dataset']
        return DataLoader(
            dataset,
            batch_size=ds_cfg.get('batch_size', 16),
            shuffle=ds_cfg.get('shuffle', False),
        )

    def evaluate_encoder(self, model, dataloader):
        """Evaluate the encoder part of the model."""
        model.eval()
        with torch.no_grad():
            z_all = []
            for batch in dataloader:
                z = model.forward_cv(batch['data'])
                z_all.append(z)
        return {"z": torch.cat(z_all, dim=0)}

    @launchlogger
    def launch(self) -> int:
        """
        Execute the :class:`EvaluateEncoder` class and its `.launch()` method.
        """

        fu.log(f'## BioBB Model Evaluator ##', self.out_log)

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        # Start Pipeline

        # load the model
        fu.log(f'Load model from {os.path.abspath(self.io_dict["in"]["input_model_pth_path"])}', self.out_log)
        model = self.load_model()

        # load the dataset
        fu.log(f'Load dataset from {os.path.abspath(self.io_dict["in"]["input_dataset_pt_path"])}', self.out_log)
        dataset = self.load_dataset()
        
        # create the dataloader
        fu.log(f'Start evaluating...', self.out_log)
        dataloader = self.create_dataloader(dataset)

        # evaluate the model
        results = self.evaluate_encoder(model, dataloader)
        
        # Save the results
        np.savez_compressed(self.io_dict["out"]["output_results_npz_path"], **results)
        fu.log(f'Evaluation Results saved to {os.path.abspath(self.io_dict["out"]["output_results_npz_path"])}', self.out_log)
        fu.log(f'File size: {get_size(self.io_dict["out"]["output_results_npz_path"])}', self.out_log)

        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)

        return 0 
        
def evaluateEncoder(
    properties: dict,
    input_model_pth_path: str,
    input_dataset_pt_path: str,
    output_results_npz_path: str,
) -> int:
    """
    Execute the :class:`EvaluateEncoder <EvaluateEncoder.EvaluateEncoder>` class and
    execute the :meth:`launch() <EvaluateEncoder.evaluateEncoder.launch>` method.
    """
    return EvaluateEncoder(
        input_model_pth_path=input_model_pth_path,
        input_dataset_pt_path=input_dataset_pt_path,
        output_results_npz_path=output_results_npz_path,
        properties=properties,
    ).launch()

evaluateEncoder.__doc__ = EvaluateEncoder.__doc__

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
        help="Additional properties for the MDFeaturizer object.",
    )

    args = parser.parse_args()
    config = args.config if args.config else None
    properties = settings.ConfReader(config=config).get_prop_dic()

    evaluateEncoder(
        input_model_pth_path=args.input_model_pth_path,
        input_dataset_pt_path=args.input_dataset_pt_path,
        output_results_npz_path=args.output_results_npz_path,
        properties=properties,
    )

if __name__ == "__main__":
    main()