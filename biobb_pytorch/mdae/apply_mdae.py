#!/usr/bin/env python3

"""Module containing the ApplyMDAE class and the command line interface."""

import argparse
import time
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from biobb_common.configuration import settings
from biobb_common.generic.biobb_object import BiobbObject
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger

from biobb_pytorch.mdae.common import (
    execute_model,
    format_time,
    human_readable_file_size,
    ndarray_denormalization,
    ndarray_normalization,
)
from biobb_pytorch.mdae.mdae import MDAE


class ApplyMDAE(BiobbObject):
    """
    | biobb_pytorch ApplyMDAE
    | Apply a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
    | Apply a Molecular Dynamics AutoEncoder (MDAE) PyTorch model, the resulting denoised molecular dynamics or the reduced the dimensionality of molecular dynamics data can be used to analyze the dynamic properties of the system.

    Args:
        input_data_npy_path (str): Path to the input data file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/data/mdae/train_mdae_traj.npy>`_. Accepted formats: npy (edam:format_4003).
        input_model_pth_path (str): Path to the input model file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        output_reconstructed_data_npy_path (str): Path to the output reconstructed data file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_reconstructed_data.npy>`_. Accepted formats: npy (edam:format_4003).
        output_latent_space_npy_path (str) (Optional): Path to the reduced dimensionality file. File type: output.  `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_latent_space.npy>`_. Accepted formats: npy (edam:format_4003).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **batch_size** (*int*) - (1) number of samples/frames per batch.
            * **latent_dimensions** (*int*) - (2) min dimensionality of the latent space.
            * **num_layers** (*int*) - (4) number of layers in the encoder/decoder (4 to encode and 4 to decode).
            * **input_dimensions** (*int*) - (None) input dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates).
            * **output_dimensions** (*int*) - (None) output dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates).

    Examples:
        This is a use case of how to use the building block from Python::

            from biobb_pytorch.mdae.apply_mdae import ApplyMDAE
            prop = {
                'latent_dimensions': 2,
                'num_layers': 4
            }
            ApplyMDAE(input_data_npy_path='/path/to/myInputData.npy',
                      output_reconstructed_data_npy_path='/path/to/newReconstructedData.npz',
                      input_model_pth_path='/path/to/oldModel.pth',
                      properties=prop).launch()

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
        input_data_npy_path: str,
        input_model_pth_path: str,
        output_reconstructed_data_npy_path: str,
        output_latent_space_npy_path: Optional[str] = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> None:
        properties = properties or {}

        # Call parent class constructor
        super().__init__(properties)
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = {
            "in": {
                "input_data_npy_path": input_data_npy_path,
                "input_model_pth_path": input_model_pth_path,
            },
            "out": {
                "output_reconstructed_data_npy_path": output_reconstructed_data_npy_path,
                "output_latent_space_npy_path": output_latent_space_npy_path,
            },
        }

        # Properties specific for BB
        self.batch_size: int = int(
            properties.get("batch_size", 1)
        )  # number of samples/frames per batch
        self.latent_dimensions: int = int(
            properties.get("latent_dimensions", 2)
        )  # min dimensionality of the latent space
        self.num_layers: int = int(
            properties.get("num_layers", 4)
        )  # number of layers in the encoder/decoder (4 to encode and 4 to decode)

        # Input data section
        input_raw_data = np.load(self.io_dict["in"]["input_data_npy_path"])
        # Reshape the input data to be a 2D array and normalization
        input_reshaped_data: np.ndarray = np.reshape(
            input_raw_data,
            (len(input_raw_data), input_raw_data.shape[1] * input_raw_data.shape[2]),
        )
        # Normalization of the input data
        self.input_data_max_values: np.ndarray = np.max(input_reshaped_data, axis=0)
        self.input_data_min_values: np.ndarray = np.min(input_reshaped_data, axis=0)
        input_data: np.ndarray = ndarray_normalization(
            input_reshaped_data,
            max_values=self.input_data_max_values,
            min_values=self.input_data_min_values,
        )
        self.input_dimensions: int = (
            int(properties["input_dimensions"])
            if properties.get("input_dimensions")
            else input_data.shape[1]
        )  # input dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)
        self.output_dimensions: int = (
            int(properties["output_dimensions"])
            if properties.get("output_dimensions")
            else self.input_dimensions
        )  # output dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

        data_tensor = torch.FloatTensor(input_data)
        tensor_dataset = torch.utils.data.TensorDataset(data_tensor)
        self.data_loader = torch.utils.data.DataLoader(
            tensor_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.model = MDAE(
            input_dimensions=self.input_dimensions,
            num_layers=self.num_layers,
            latent_dimensions=self.latent_dimensions,
        )
        self.model.load_state_dict(
            torch.load(
                self.io_dict["in"]["input_model_pth_path"],
                map_location=self.model.device,
            )
        )

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`ApplyMDAE <mdae.apply_mdae.ApplyMDAE>` object."""

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        fu.log(
            f"Applying MDAE model reducing dimensionality from {self.input_dimensions} to {self.latent_dimensions} and reconstructing.",
            self.out_log,
        )
        latent_space, reconstructed_data = self.apply_model(self.data_loader)
        denormalized_reconstructed_data = ndarray_denormalization(
            reconstructed_data, self.input_data_max_values, self.input_data_min_values
        )
        reshaped_reconstructed_data = np.reshape(
            denormalized_reconstructed_data,
            (len(denormalized_reconstructed_data), -1, 3),
        )
        np.save(
            self.stage_io_dict["out"]["output_reconstructed_data_npy_path"],
            np.array(reshaped_reconstructed_data),
        )
        fu.log(
            f'Saving reconstructed data to: {self.stage_io_dict["out"]["output_reconstructed_data_npy_path"]}',
            self.out_log,
        )
        fu.log(
            f'  File size: {human_readable_file_size(self.stage_io_dict["out"]["output_reconstructed_data_npy_path"])}',
            self.out_log,
        )

        if self.stage_io_dict["out"].get("output_latent_space_npy_path"):
            np.save(
                self.stage_io_dict["out"]["output_latent_space_npy_path"],
                np.array(latent_space),
            )
            fu.log(
                f'Saving latent space to: {self.stage_io_dict["out"]["output_latent_space_npy_path"]}',
                self.out_log,
            )
            fu.log(
                f'  File size: {human_readable_file_size(self.stage_io_dict["out"]["output_latent_space_npy_path"])}',
                self.out_log,
            )

        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)
        return 0

    def apply_model(
        self, dataloader: torch.utils.data.DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model.to(self.model.device)
        start_time: float = time.time()
        fu.log("Applying model:", self.out_log)
        fu.log(f"  Device: {self.model.device}", self.out_log)
        fu.log(
            f"  Input file: {self.stage_io_dict['in']['input_data_npy_path']}",
            self.out_log,
        )
        fu.log(
            f"    File size: {human_readable_file_size(self.stage_io_dict['in']['input_data_npy_path'])}",
            self.out_log,
        )
        fu.log(
            f"  Number of atoms: {int(len(next(iter(dataloader))[0][0])/3)}",
            self.out_log,
        )
        fu.log(
            f"  Number of frames: {int(len(dataloader) * (dataloader.batch_size or 1))}",
            self.out_log,
        )  # type: ignore
        fu.log(f"  Batch size: {self.batch_size}", self.out_log)
        fu.log(f"  Number of layers: {self.num_layers}", self.out_log)
        fu.log(f"  Input dimensions: {self.input_dimensions}", self.out_log)
        fu.log(f"  Latent dimensions: {self.latent_dimensions}", self.out_log)

        execution_tuple = execute_model(
            self.model, dataloader, self.input_dimensions, self.latent_dimensions
        )[1:]

        fu.log(
            f"  Execution time: {format_time(time.time() - start_time)}", self.out_log
        )
        return execution_tuple


def applyMDAE(
    input_data_npy_path: str,
    input_model_pth_path: str,
    output_reconstructed_data_npy_path: str,
    output_latent_space_npy_path: Optional[str] = None,
    properties: Optional[dict] = None,
    **kwargs,
) -> int:
    """Execute the :class:`ApplyMDAE <mdae.apply_mdae.ApplyMDAE>` class and
    execute the :meth:`launch() <mdae.apply_mdae.ApplyMDAE.launch>` method."""

    return ApplyMDAE(
        input_data_npy_path=input_data_npy_path,
        input_model_pth_path=input_model_pth_path,
        output_reconstructed_data_npy_path=output_reconstructed_data_npy_path,
        output_latent_space_npy_path=output_latent_space_npy_path,
        properties=properties,
        **kwargs,
    ).launch()


def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(
        description="Apply a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999),
    )
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        help="This file can be a YAML file, JSON file or JSON string",
    )

    # Specific args of each building block
    required_args = parser.add_argument_group("required arguments")

    required_args.add_argument(
        "--input_data_npy_path", required=True, help="Path to the input data file."
    )
    required_args.add_argument(
        "--input_model_pth_path", required=True, help="Path to the input model file."
    )
    required_args.add_argument(
        "--output_reconstructed_data_npy_path",
        required=True,
        help="Path to the output reconstructed data file.",
    )
    parser.add_argument(
        "--output_latent_space_npy_path",
        required=False,
        help="Path to the reduced dimensionality file.",
    )
    parser.add_argument(
        "--properties",
        required=False,
        help="Additional properties for the MDAE object.",
    )
    args = parser.parse_args()
    config = args.config if args.config else None
    properties = settings.ConfReader(config=config).get_prop_dic()

    applyMDAE(
        input_data_npy_path=args.input_data_npy_path,
        input_model_pth_path=args.input_model_pth_path,
        output_reconstructed_data_npy_path=args.output_reconstructed_data_npy_path,
        output_latent_space_npy_path=args.output_latent_space_npy_path,
        properties=properties,
    )


if __name__ == "__main__":
    main()
