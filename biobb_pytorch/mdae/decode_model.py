import torch
from torch.utils.data import DataLoader
import os
from biobb_common.tools.file_utils import launchlogger
from biobb_common.tools import file_utils as fu
from biobb_pytorch.mdae.utils.log_utils import get_size
from biobb_common.generic.biobb_object import BiobbObject
import numpy as np


class EvaluateDecoder(BiobbObject):
    """
    | biobb_pytorch evaluateDecoder
    | Evaluates a PyTorch autoencoder from the given properties.
    | Evaluates a PyTorch autoencoder from the given properties.

    Args:
        input_model_pth_path (str): Path to the trained model file whose decoder will be used. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        input_dataset_npy_path (str): Path to the input latent variables file in NumPy format (e.g. encoded 'z'). File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.npy>`_. Accepted formats: npy (edam:format_2333).
        output_results_npz_path (str): Path to the output reconstructed data file (compressed NumPy archive, typically containing 'xhat'). File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_results.npz>`_. Accepted formats: npz (edam:format_2333).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **Dataset** (*dict*) - ({}) DataLoader options (e.g. batch_size, shuffle) for batching the latent variables.

    Examples:
        This example shows how to use the EvaluateDecoder class to evaluate a PyTorch autoencoder model::

            from biobb_pytorch.mdae.decode_model import evaluateDecoder

            input_model_pth_path='input_model.pth'
            input_dataset_npy_path='input_dataset.npy'
            output_results_npz_path='output_results.npz'

            prop={
                'Dataset': {
                    'batch_size': 32
                }
            }

            evaluateDecoder(input_model_pth_path=input_model.pth,
                    input_dataset_npy_path=input_dataset.npy,
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
        input_dataset_npy_path: str,
        output_results_npz_path: str,
        properties: dict,
        **kwargs,
    ) -> None:

        properties = properties or {}

        super().__init__(properties)

        self.input_model_pth_path = input_model_pth_path
        self.input_dataset_npy_path = input_dataset_npy_path
        self.output_results_npz_path = output_results_npz_path
        self.properties = properties.copy()
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = {
            "in": {
                "input_model_pth_path": input_model_pth_path,
                "input_dataset_npy_path": input_dataset_npy_path,
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
        return torch.load(self.io_dict["in"]["input_model_pth_path"],
                          weights_only=False)

    def load_dataset(self):
        dataset = torch.tensor(np.load(self.io_dict["in"]["input_dataset_npy_path"]))
        return dataset.float()

    def create_dataloader(self, dataset):
        ds_cfg = self.properties['Dataset']
        return DataLoader(
            dataset,
            batch_size=ds_cfg.get('batch_size', 16),
            shuffle=ds_cfg.get('shuffle', False),
        )

    def evaluate_decoder(self, model, dataloader):
        """Evaluate the decoder part of the model."""
        model.eval()
        with torch.no_grad():
            all_reconstructions = []
            for batch in dataloader:
                z = model.decode(batch)
                all_reconstructions.append(z)
        return {"xhat": torch.cat(all_reconstructions, dim=0)}

    @launchlogger
    def launch(self) -> int:
        """
        Execute the :class:`EvaluateDecoder` class and its `.launch()` method.
        """

        fu.log('## BioBB Model Evaluator ##', self.out_log)

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        # Start Pipeline

        # load the model
        fu.log(f'Load model from {os.path.abspath(self.io_dict["in"]["input_model_pth_path"])}', self.out_log)
        model = self.load_model()

        # load the dataset
        fu.log(f'Load dataset from {os.path.abspath(self.io_dict["in"]["input_dataset_npy_path"])}', self.out_log)
        dataset = self.load_dataset()

        # create the dataloader
        fu.log('Start evaluating...', self.out_log)
        dataloader = self.create_dataloader(dataset)

        # evaluate the model
        results = self.evaluate_decoder(model, dataloader)

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


def evaluateDecoder(
    properties: dict,
    input_model_pth_path: str,
    input_dataset_npy_path: str,
    output_results_npz_path: str,
    **kwargs,
) -> int:
    """Create the :class:`EvaluateDecoder <EvaluateDecoder.EvaluateDecoder>` class and
    execute the :meth:`launch() <EvaluateDecoder.evaluateDecoder.launch>` method."""
    return EvaluateDecoder(**dict(locals())).launch()


evaluateDecoder.__doc__ = EvaluateDecoder.__doc__
main = EvaluateDecoder.get_main(evaluateDecoder, "Evaluates a PyTorch autoencoder from the given properties.")

if __name__ == "__main__":
    main()
