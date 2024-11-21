#!/usr/bin/env python3

"""Module containing the TrainMDAE class and the command line interface."""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from biobb_common.configuration import settings
from biobb_common.generic.biobb_object import BiobbObject
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from biobb_pytorch.mdae.common import (
    execute_model,
    format_time,
    get_loss_function,
    get_optimizer_function,
    human_readable_file_size,
    ndarray_denormalization,
    ndarray_normalization,
)
from biobb_pytorch.mdae.mdae import MDAE


class TrainMDAE(BiobbObject):
    """
    | biobb_pytorch TrainMDAE
    | Train a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
    | Train a Molecular Dynamics AutoEncoder (MDAE) PyTorch model, the resulting Auto-associative Neural Network (AANN) can be applied to reduce the dimensionality of molecular dynamics data and analyze the dynamic properties of the system.

    Args:
        input_train_npy_path (str): Path to the input train data file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/data/mdae/train_mdae_traj.npy>`_. Accepted formats: npy (edam:format_4003).
        output_model_pth_path (str): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        input_model_pth_path (str) (Optional): Path to the input model file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        output_train_data_npz_path (str) (Optional): Path to the output train data file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_train_data.npz>`_. Accepted formats: npz (edam:format_4003).
        output_performance_npz_path (str) (Optional): Path to the output performance file. File type: output.  `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_performance.npz>`_. Accepted formats: npz (edam:format_4003).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **latent_dimensions** (*int*) - (2) min dimensionality of the latent space.
            * **num_layers** (*int*) - (4) number of layers in the encoder/decoder (4 to encode and 4 to decode).
            * **num_epochs** (*int*) - (100) number of epochs (iterations of whole dataset) for training.
            * **lr** (*float*) - (0.0001) learning rate.
            * **lr_step_size** (*int*) - (100) Period of learning rate decay.
            * **gamma** (*float*) - (0.1) Multiplicative factor of learning rate decay.
            * **checkpoint_interval** (*int*) - (25) number of epochs interval to save model checkpoints o 0 to disable.
            * **output_checkpoint_prefix** (*str*) - ("checkpoint_epoch") prefix for the checkpoint files.
            * **partition** (*float*) - (0.8) 0.8 = 80% partition of the data for training and validation.
            * **batch_size** (*int*) - (1) number of samples/frames per batch.
            * **log_interval** (*int*) - (10) number of epochs interval to log the training progress.
            * **input_dimensions** (*int*) - (None) input dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates).
            * **output_dimensions** (*int*) - (None) output dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates).
            * **loss_function** (*str*) - ("MSELoss") Loss function to be used. Values: MSELoss, L1Loss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, CTCLoss, NLLLoss, KLDivLoss, PoissonNLLLoss, NLLLoss2d, CosineEmbeddingLoss, HingeEmbeddingLoss, MarginRankingLoss, MultiLabelMarginLoss, MultiLabelSoftMarginLoss, MultiMarginLoss, TripletMarginLoss, HuberLoss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, MarginRankingLoss, HingeEmbeddingLoss, CTCLoss, NLLLoss, PoissonNLLLoss, KLDivLoss, CrossEntropyLoss, BCEWithLogitsLoss, BCELoss, SmoothL1Loss, L1Loss, MSELoss.
            * **optimizer** (*str*) - ("Adam") Optimizer algorithm to be used. Values: Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD.
            * **seed** (*int*) - (None) Random seed for reproducibility.

    Examples:
        This is a use case of how to use the building block from Python::

            from biobb_pytorch.mdae.train_mdae import trainMDAE

            prop = {
                'latent_dimensions': 2,
                'num_layers': 4,
                'num_epochs': 100,
                'lr': 0.0001,
                'checkpoint_interval': 25,
                'partition': 0.8,
                'batch_size': 1,
                'log_interval': 10,
                'input_dimensions': 3,
                'output_dimensions': 3,
                'loss_function': 'MSELoss',
                'optimizer': 'Adam'
            }

            trainMDAE(input_train_npy_path='/path/to/myInputData.npy',
                      output_model_pth_path='/path/to/newModel.pth',
                      input_model_pth_path='/path/to/oldModel.pth',
                      output_train_data_npz_path='/path/to/newTrainData.npz',
                      output_performance_npz_path='/path/to/newPerformance.npz',
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
        input_train_npy_path: str,
        output_model_pth_path: str,
        input_model_pth_path: Optional[str] = None,
        output_train_data_npz_path: Optional[
            str
        ] = None,  # npz of  train_losses, valid_losses
        output_performance_npz_path: Optional[
            str
        ] = None,  # npz of  evaluate_losses, latent_space, reconstructed_data
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
                "input_train_npy_path": input_train_npy_path,
                "input_model_pth_path": input_model_pth_path,
            },
            "out": {
                "output_model_pth_path": output_model_pth_path,
                "output_train_data_npz_path": output_train_data_npz_path,
                "output_performance_npz_path": output_performance_npz_path,
            },
        }

        # Properties specific for BB
        self.latent_dimensions: int = int(
            properties.get("latent_dimensions", 2)
        )  # min dimensionality of the latent space
        self.num_layers: int = int(
            properties.get("num_layers", 4)
        )  # number of layers in the encoder/decoder (4 to encode and 4 to decode)
        self.num_epochs: int = int(
            properties.get("num_epochs", 100)
        )  # number of epochs (iterations of whole dataset) for training
        self.lr: float = float(properties.get("lr", 0.0001))  # learning rate
        self.lr_step_size: int = int(
            properties.get("lr_step_size", 100)
        )  # Period of learning rate decay
        self.gamma: float = float(
            properties.get("gamma", 0.1)
        )  # Multiplicative factor of learning rate decay
        self.checkpoint_interval: int = int(
            properties.get("checkpoint_interval", 25)
        )  # number of epochs interval to save model checkpoints o 0 to disable
        self.output_checkpoint_prefix: str = properties.get(
            "output_checkpoint_prefix", "checkpoint_epoch_"
        )  # prefix for the checkpoint files,
        self.partition: float = float(
            properties.get("partition", 0.8)
        )  # 0.8 = 80% partition of the data for training and validation
        self.seed: Optional[int] = (
            int(properties.get("seed", "42")) if properties.get("seed", None) else None
        )  # Random seed for reproducibility
        self.batch_size: int = int(
            properties.get("batch_size", 1)
        )  # number of samples/frames per batch
        self.log_interval: int = int(
            properties.get("log_interval", 10)
        )  # number of epochs interval to log the training progress

        # Input data section
        input_raw_data = np.load(self.io_dict["in"]["input_train_npy_path"])
        # Reshape the input data to be a 2D array and normalization
        input_train_reshaped_data: np.ndarray = np.reshape(
            input_raw_data,
            (len(input_raw_data), input_raw_data.shape[1] * input_raw_data.shape[2]),
        )
        # Normalization of the input data
        self.input_train_data_max_values: np.ndarray = np.max(
            input_train_reshaped_data, axis=0
        )
        self.input_train_data_min_values: np.ndarray = np.min(
            input_train_reshaped_data, axis=0
        )
        input_train_data: np.ndarray = ndarray_normalization(
            input_train_reshaped_data,
            max_values=self.input_train_data_max_values,
            min_values=self.input_train_data_min_values,
        )

        self.input_dimensions: int = (
            int(properties["input_dimensions"])
            if properties.get("input_dimensions")
            else input_train_data.shape[1]
        )  # input dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)
        self.output_dimensions: int = (
            int(properties["output_dimensions"])
            if properties.get("output_dimensions")
            else self.input_dimensions
        )  # output dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

        # Select the data for training and validation steps
        index_train_data = int(self.partition * input_train_data.shape[0])
        index_validation_data = int((1 - self.partition) * input_train_data.shape[0])
        train_tensor = torch.FloatTensor(input_train_data[:index_train_data, :])
        validation_tensor = torch.FloatTensor(
            input_train_data[-index_validation_data:, :]
        )
        performance_tensor = torch.FloatTensor(input_train_data)
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        validation_dataset = torch.utils.data.TensorDataset(validation_tensor)
        performance_dataset = torch.utils.data.TensorDataset(performance_tensor)

        # Seed
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self.train_dataloader: torch.utils.data.DataLoader = (
            torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
            )
        )
        self.validation_dataloader: torch.utils.data.DataLoader = (
            torch.utils.data.DataLoader(
                dataset=validation_dataset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=False,
            )
        )
        self.performance_dataloader: torch.utils.data.DataLoader = (
            torch.utils.data.DataLoader(
                dataset=performance_dataset,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
            )
        )

        # Create the model
        self.model = MDAE(
            input_dimensions=self.input_dimensions,
            num_layers=self.num_layers,
            latent_dimensions=self.latent_dimensions,
        )
        if self.io_dict["in"]["input_model_pth_path"]:
            self.model.load_state_dict(
                torch.load(
                    self.io_dict["in"]["input_model_pth_path"],
                    map_location=self.model.device,
                )
            )

        # Define loss function and optimizer algorithm
        loss_function_str: str = properties.get("loss_function", "")
        try:
            self.loss_function: torch.nn.modules.loss._Loss = get_loss_function(
                loss_function_str
            )()
            fu.log(f"Using loss function: {self.loss_function}", self.out_log)
        except ValueError:
            fu.log(f"Invalid loss function: {loss_function_str}", self.out_log)
            fu.log("Using default loss function: MSELoss", self.out_log)
            self.loss_function = torch.nn.MSELoss()

        optimizer_str: str = properties.get("optimizer", "")
        try:
            self.optimizer = get_optimizer_function(optimizer_str)(
                self.model.parameters(), lr=self.lr
            )
            fu.log(f"Using optimizer: {self.optimizer}", self.out_log)
        except ValueError:
            fu.log(f"Invalid optimizer: {optimizer_str}", self.out_log)
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`TrainMDAE <mdae.train_mdae.TrainMDAE>` object."""

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        # Train the model
        train_losses, validation_losses, best_model, best_model_epoch = (
            self.train_model()
        )
        if self.stage_io_dict["out"].get("output_train_data_npz_path"):
            np.savez(
                self.stage_io_dict["out"]["output_train_data_npz_path"],
                train_losses=np.array(train_losses),
                validation_losses=np.array(validation_losses),
            )
            fu.log(
                f'Saving train data to: {self.stage_io_dict["out"]["output_train_data_npz_path"]}',
                self.out_log,
            )
            fu.log(
                f'  File size: {human_readable_file_size(self.stage_io_dict["out"]["output_train_data_npz_path"])}',
                self.out_log,
            )

        # Evaluate the model
        if self.stage_io_dict["out"].get("output_performance_npz_path"):
            evaluate_losses, latent_space, reconstructed_data = self.evaluate_model(
                self.performance_dataloader, self.loss_function
            )
            denormalized_reconstructed_data = ndarray_denormalization(
                reconstructed_data,
                self.input_train_data_max_values,
                self.input_train_data_min_values,
            )
            reshaped_reconstructed_data = np.reshape(
                denormalized_reconstructed_data,
                (len(denormalized_reconstructed_data), -1, 3),
            )
            np.savez(
                self.stage_io_dict["out"]["output_performance_npz_path"],
                evaluate_losses=np.array(evaluate_losses),
                latent_space=np.array(latent_space),
                denormalized_reconstructed_data=np.array(reshaped_reconstructed_data),
            )
            fu.log(
                f'Saving evaluation data to: {self.stage_io_dict["out"]["output_performance_npz_path"]}',
                self.out_log,
            )
            fu.log(
                f'  File size: {human_readable_file_size(self.stage_io_dict["out"]["output_performance_npz_path"])}',
                self.out_log,
            )

        # Save the model
        torch.save(best_model, self.stage_io_dict["out"]["output_model_pth_path"])
        fu.log(
            f'Saving best model to: {self.stage_io_dict["out"]["output_model_pth_path"]}',
            self.out_log,
        )
        fu.log(f"  Best model epoch: {best_model_epoch}", self.out_log)
        fu.log(
            f'  File size: {human_readable_file_size(self.stage_io_dict["out"]["output_model_pth_path"])}',
            self.out_log,
        )

        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)
        return 0

    def train_model(self) -> tuple[list[float], list[float], dict, int]:
        self.model.to(self.model.device)
        train_losses: list[float] = []
        validation_losses: list[float] = []
        best_valid_loss: float = float("inf")  # Initialize best valid loss to infinity

        start_time: float = time.time()
        fu.log("Start Training:", self.out_log)
        fu.log(f"  Device: {self.model.device}", self.out_log)
        fu.log(
            f"  Train input file: {self.stage_io_dict['in']['input_train_npy_path']}",
            self.out_log,
        )
        fu.log(
            f"    File size: {human_readable_file_size(self.stage_io_dict['in']['input_train_npy_path'])}",
            self.out_log,
        )
        fu.log(
            f"  Number of atoms: {int(len(next(iter(self.train_dataloader))[0][0])/3)}",
            self.out_log,
        )
        fu.log(
            f"  Number of frames for training: {len(self.train_dataloader)*self.train_dataloader.batch_size} Total number of frames: {int((len(self.train_dataloader)*self.train_dataloader.batch_size)/self.partition) if self.partition is not None else 'Unknown'}",
            self.out_log,
        )  # type: ignore
        fu.log(f"  Number of epochs: {self.num_epochs}", self.out_log)
        fu.log(f"  Partition: {self.partition}", self.out_log)
        fu.log(f"  Batch size: {self.batch_size}", self.out_log)
        fu.log(f"  Learning rate: {self.lr}", self.out_log)
        fu.log(f"  Learning rate step size: {self.lr_step_size}", self.out_log)
        fu.log(f"  Learning rate gamma: {self.gamma}", self.out_log)
        fu.log(f"  Number of layers: {self.num_layers}", self.out_log)
        fu.log(f"  Input dimensions: {self.input_dimensions}", self.out_log)
        fu.log(f"  Latent dimensions: {self.latent_dimensions}", self.out_log)
        fu.log(
            f"  Loss function: {str(self.loss_function).split('(')[0]}", self.out_log
        )
        fu.log(f"  Optimizer: {str(self.optimizer).split('(')[0]}", self.out_log)
        fu.log(f"  Seed: {self.seed}", self.out_log)
        fu.log(f"  Checkpoint interval: {self.checkpoint_interval}", self.out_log)
        fu.log(f"  Log interval: {self.log_interval}\n", self.out_log)

        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.lr_step_size, gamma=self.gamma
        )
        for epoch_index in range(self.num_epochs):
            loop_start_time: float = time.time()

            # Training & validation step
            avg_train_loss, avg_validation_loss = self.training_step(
                self.train_dataloader, self.optimizer, self.loss_function
            )
            train_losses.append(avg_train_loss)
            validation_losses.append(avg_validation_loss)

            # Logging
            if self.log_interval and (
                epoch_index % self.log_interval == 0
                or epoch_index == self.num_epochs - 1
            ):
                epoch_time: float = time.time() - loop_start_time
                fu.log(
                    f'{"Epoch":>4} {epoch_index+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.3f}, Validation Loss: {avg_validation_loss:.3f}, LR: {scheduler.get_last_lr()[0]:.5f}, Duration: {format_time(epoch_time)}, ETA: {format_time((self.num_epochs-(epoch_index+1))*epoch_time)}',
                    self.out_log,
                )
                loop_start_time = time.time()

            # Save checkpoint
            if self.checkpoint_interval and (
                epoch_index % self.checkpoint_interval == 0
                or epoch_index == self.num_epochs - 1
            ):
                checkpoint_path = str(
                    Path(self.stage_io_dict.get("unique_dir", "")).joinpath(
                        f"{self.output_checkpoint_prefix}_{epoch_index}.pth"
                    )
                )
                fu.log(f'{"Saving: ":>4} {checkpoint_path}', self.out_log)
                torch.save(self.model.state_dict(), checkpoint_path)

            # Update learning rate
            scheduler.step()

            # Save best model
            if avg_validation_loss < best_valid_loss:
                best_valid_loss = avg_validation_loss
                best_model: dict = self.model.state_dict()
                best_model_epoch: int = epoch_index

        fu.log(
            f"End Training, total time: {format_time((time.time() - start_time))}",
            self.out_log,
        )

        return train_losses, validation_losses, best_model, best_model_epoch

    def training_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        loss_function: torch.nn.modules.loss._Loss,
    ) -> tuple[float, float]:
        self.model.train()
        train_losses: list[float] = []
        for data in dataloader:
            data = data[0].to(self.model.device)
            _, output = self.model(data)
            loss = loss_function(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        self.model.eval()
        valid_losses: list[float] = []
        with torch.no_grad():
            for data in dataloader:
                data = data[0].to(self.model.device)
                _, output = self.model(data)
                loss = loss_function(output, data)
                valid_losses.append(loss.item())

        return float(np.mean(train_losses)), float(
            torch.mean(torch.tensor(valid_losses))
        )

    def evaluate_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_function: torch.nn.modules.loss._Loss,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        return execute_model(
            self.model,
            dataloader,
            self.input_dimensions,
            self.latent_dimensions,
            loss_function,
        )


def trainMDAE(
    input_train_npy_path: str,
    output_model_pth_path: str,
    input_model_pth_path: Optional[str] = None,
    output_train_data_npz_path: Optional[str] = None,
    output_performance_npz_path: Optional[str] = None,
    properties: Optional[dict] = None,
    **kwargs,
) -> int:
    """Execute the :class:`TrainMDAE <mdae.train_mdae.TrainMDAE>` class and
    execute the :meth:`launch() <mdae.train_mdae.TrainMDAE.launch>` method."""

    return TrainMDAE(
        input_train_npy_path=input_train_npy_path,
        output_model_pth_path=output_model_pth_path,
        input_model_pth_path=input_model_pth_path,
        output_train_data_npz_path=output_train_data_npz_path,
        output_performance_npz_path=output_performance_npz_path,
        properties=properties,
        **kwargs,
    ).launch()


def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(
        description="Train a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.",
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
        "--input_train_npy_path",
        required=True,
        help="Path to the input train data file. Accepted formats: npy.",
    )
    required_args.add_argument(
        "--output_model_pth_path",
        required=True,
        help="Path to the output model file. Accepted formats: pth.",
    )
    parser.add_argument(
        "--input_model_pth_path",
        required=False,
        help="Path to the input model file. Accepted formats: pth.",
    )
    parser.add_argument(
        "--output_train_data_npz_path",
        required=False,
        help="Path to the output train data file. Accepted formats: npz.",
    )
    parser.add_argument(
        "--output_performance_npz_path",
        required=False,
        help="Path to the output performance file. Accepted formats: npz.",
    )
    parser.add_argument(
        "--properties",
        required=False,
        help="Additional properties for the MDAE object.",
    )
    args = parser.parse_args()
    config = args.config if args.config else None
    properties = settings.ConfReader(config=config).get_prop_dic()

    trainMDAE(
        input_train_npy_path=args.input_train_npy_path,
        output_model_pth_path=args.output_model_pth_path,
        input_model_pth_path=args.input_model_pth_path,
        output_train_data_npz_path=args.output_train_data_npz_path,
        output_performance_npz_path=args.output_performance_npz_path,
        properties=properties,
    )


if __name__ == "__main__":
    main()
