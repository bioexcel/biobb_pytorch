import os
import argparse
import torch
from torch.utils.data import DataLoader
import importlib
from typing import Dict, Any, Type
from typing import Optional
from biobb_common.configuration import settings
from biobb_common.tools.file_utils import launchlogger
from biobb_common.tools import file_utils as fu
from biobb_pytorch.mdae.utils.log_utils import get_size
from biobb_common.generic.biobb_object import BiobbObject
from mlcolvar.data import DictDataset
import numpy as np

class EvaluateModel(BiobbObject):
    """
    | biobb_pytorch EvaluateModel
    | Evaluates a PyTorch autoencoder from the given properties.
    | Evaluates a PyTorch autoencoder from the given properties.

    Args:
        input_model_pth_path (str): Path to the input model file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        input_dataset_pt_path (str): Path to the input dataset file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pt>`_. Accepted formats: pt (edam:format_2333).
        output_results_npz_path (str): Path to the output results file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_results.npz>`_. Accepted formats: npz (edam:format_2333).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **Dataset** (*dict*) - ({}) Dataset options from mlcolvar.

    Examples:
        This example shows how to use the EvaluateModel class to evaluate a PyTorch autoencoder model::

            from biobb_pytorch.mdae.evaluate_model import evaluateModel
            
            input_model_pth_path='input_model.pth'
            input_dataset_pt_path='input_dataset.pt'
            output_results_npz_path='output_results.npz'

            prop={
                'Dataset': {
                    'batch_size': 32
                }
            }
            
            evaluateModel(input_model_pth_path=input_model.pth,
                    input_dataset_pt_path=input_dataset.pt,
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
                
    def evaluate_full_model(self, model, dataloader):
        """Evaluate the model on the data, computing average loss and collecting output variables."""
        
        output_variables = model.eval_variables
        all_results = []
        all_losses = []
        result_dict = {}
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                result = model.evaluate_model(batch, batch_idx)
                # Note: Consider replacing with model.validation_step(batch, batch_idx) or 
                # model.loss_fn(model(batch['data']), batch['data']) for eval-specific loss
                batch_loss = model.training_step(batch, batch_idx)
                all_results.append(result)
                all_losses.append(batch_loss.item())  # Use .item() to get scalar
        
        # After all batches, collect per variable (assuming result is list/tuple of tensors)
        for i, var in enumerate(output_variables):
            var_results = [res[i] for res in all_results]
            result_dict[var] = torch.cat(var_results) if var_results else torch.tensor([])  # Concat if tensors
        
        # Average loss (use np.mean for simplicity with list of scalars)
        avg_loss = np.mean(all_losses) if all_losses else 0.0
        
        # Add to dictionary
        result_dict['eval_loss'] = avg_loss
        
        # Optional: Convert tensors to NumPy arrays for saving to .npz
        for key in result_dict:
            if torch.is_tensor(result_dict[key]):
                result_dict[key] = result_dict[key].numpy()
        
        return result_dict
    
    def evaluate_encoder(self, model, dataloader):
        """Evaluate the encoder part of the model."""
        model.eval()
        with torch.no_grad():
            all_z = []
            for batch in dataloader:
                z = model.forward_cv(batch['data'])
                all_z.append(z)
        return torch.cat(all_z, dim=0) if all_z else torch.tensor([])
    
    def evaluate_decoder(self, model, dataloader):
        """Evaluate the decoder part of the model."""
        model.eval()
        with torch.no_grad():
            all_reconstructions = []
            for batch in dataloader:
                reconstructions = model.decode(batch['data'])
                all_reconstructions.append(reconstructions)
        return torch.cat(all_reconstructions, dim=0) if all_reconstructions else torch.tensor([])
    
    @launchlogger
    def launch(self) -> int:
        """
        Execute the :class:`EvaluateModel` class and its `.launch()` method.
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
        results = self.evaluate_full_model(model, dataloader)
        
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
        
def evaluateModel(
    properties: dict,
    input_model_pth_path: str,
    input_dataset_pt_path: str,
    output_results_npz_path: str,
) -> int:
    """
    Execute the :class:`EvaluateModel <EvaluateModel.EvaluateModel>` class and
    execute the :meth:`launch() <EvaluateModel.evaluateModel.launch>` method.
    """
    return EvaluateModel(
        input_model_pth_path=input_model_pth_path,
        input_dataset_pt_path=input_dataset_pt_path,
        output_results_npz_path=output_results_npz_path,
        properties=properties,
    ).launch()

evaluateModel.__doc__ = EvaluateModel.__doc__

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

    evaluateModel(
        input_model_pth_path=args.input_model_pth_path,
        input_dataset_pt_path=args.input_dataset_pt_path,
        output_results_npz_path=args.output_results_npz_path,
        properties=properties,
    )

if __name__ == "__main__":
    main()