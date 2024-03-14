"""Module containing the MDAutoEncoder class and the command line interface."""
import torch
import numpy as np
import time
import argparse
from typing import Optional, List, Tuple, Dict
from biobb_common.generic.biobb_object import BiobbObject
from biobb_common.configuration import settings
from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_pytorch.mdae.mdae import MDAE
from biobb_pytorch.mdae.common import get_loss_function, get_optimizer_function, ndarray_normalization, ndarray_denormalization
from pathlib import Path


class TrainMDAE(BiobbObject):
    """
    | biobb_pytorch TrainMDAE
    | Train a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
    | Train a Molecular Dynamics AutoEncoder (MDAE) PyTorch model, the resulting Auto-associative Neural Network (AANN) can be applied to reduce the dimensionality of molecular dynamics data and analyze the dynamic properties of the system.

    Args:
        input_model_path (str): Path to the input model file. File type: input. `Sample file <
        properties:
            latent_dimensions (int): Minimum dimensionality of the latent space. Default: 2.
            num_layers (int): Number of layers in the encoder/decoder. Default: 4.
            num_epochs (int): Number of epochs (iterations of whole dataset) for training. Default: 100.
            lr (float): Learning rate. Default: 0.001.
            optimizer (str): Optimizer algorithm. Values: https://pytorch.org/docs/stable/optim.html#algorithms
            loss_function (str): Loss function. Values: https://pytorch.org/docs/stable/nn.html#loss-functions.
            input_dimensions (int): Input dimensions. Default: 0.
            output_dimensions (int): Output dimensions. Default: 0.
            checkpoint_interval (int): Number of epochs interval to save model checkpoints. Default: 25.
            partition (float): Partition of the data for training and validation. Default: 0.8.
            batch_size (int): Number of samples/frames per batch. Default: 1.
    """

    def __init__(self, input_train_npy_path: str,
                 output_model_pth_path: str,
                 input_model_pth_path: Optional[str],
                 output_train_data_npy_path: Optional[str],  # npy of  train_losses, valid_losses
                 output_performance_npy_path: Optional[str],  # npy of  evaluate_losses, latent_space, reconstructed_data
                 properties: Optional[Dict] = None, **kwargs) -> None:
        properties = properties or {}

        # Call parent class constructor
        super().__init__(properties)
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = {
            'in': {'input_train_npy_path': input_train_npy_path, 'input_model_pth_path': input_model_pth_path},
            'out': {'output_model_pth_path': output_model_pth_path, 'output_train_data_npy_path': output_train_data_npy_path, 'output_performance_npy_path': output_performance_npy_path}
        }

        # Properties specific for BB
        self.latent_dimensions: int = int(properties.get('latent_dimensions', 2))  # min dimensionality of the latent space
        self.num_layers: int = int(properties.get('num_layers', 4))  # number of layers in the encoder/decoder (4 to encode and 4 to decode)
        self.num_epochs: int = int(properties.get('num_epochs', 100))  # number of epochs (iterations of whole dataset) for training
        self.lr: float = float(properties.get('lr', 0.001))  # learning rate
        self.checkpoint_interval: int = int(properties.get('checkpoint_interval', 25))  # number of epochs interval to save model checkpoints o 0 to disable
        self.output_checkpoint_prefix: str = properties.get('output_checkpoint_prefix', 'checkpoint_epoch_')  # prefix for the checkpoint files,
        self.partition: int = int(properties.get('partition', 0.8))  # 0.8 = 80% partition of the data for training and validation
        self.batch_size: int = int(properties.get('batch_size', 1))  # number of samples/frames per batch
        self.log_interval: int = int(properties.get('log_interval', 10))  # number of epochs interval to log the training progress

        # Input data section
        input_raw_data = np.load(self.io_dict['in']['input_train_npy_path'])
        # Reshape the input data to be a 2D array and normalization
        input_train_reshaped_data: np.ndarray = np.reshape(input_raw_data, (len(input_raw_data), input_raw_data.shape[1]*input_raw_data.shape[2]))
        # Normalization of the input data
        self.input_train_data_max_values: np.ndarray = np.max(input_train_reshaped_data, axis=0)
        self.input_train_data_min_values: np.ndarray = np.min(input_train_reshaped_data, axis=0)
        input_train_data: np.ndarray = ndarray_normalization(input_train_reshaped_data, max_values=self.input_train_data_max_values, min_values=self.input_train_data_min_values)

        self.input_dimensions: int = int(properties['input_dimensions']) if properties.get('input_dimensions') else input_train_data.shape[1]  # input dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)
        self.output_dimensions: int = int(properties['output_dimensions']) if properties.get('output_dimensions') else self.input_dimensions  # output dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

        # Select the data for training and validation steps
        index_train_data = int(self.partition*input_train_data.shape[0])
        index_validation_data = int((1-self.partition)*input_train_data.shape[0])
        train_tensor = torch.FloatTensor(input_train_data[:index_train_data, :])
        validation_tensor = torch.FloatTensor(input_train_data[-index_validation_data:, :])
        performance_tensor = torch.FloatTensor(input_train_data)
        train_dataset = torch.data.TensorDataset(train_tensor)
        validation_dataset = torch.data.TensorDataset(validation_tensor)
        performance_dataset = torch.data.TensorDataset(performance_tensor)

        self.train_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                                                         batch_size=self.batch_size,
                                                                                         drop_last=True,
                                                                                         shuffle=True)
        self.validation_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                                                              batch_size=self.batch_size,
                                                                                              drop_last=True,
                                                                                              shuffle=False)
        self.performance_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset=performance_dataset,
                                                                                               batch_size=self.batch_size,
                                                                                               drop_last=False,
                                                                                               shuffle=False)

        # Create the model
        self.model = MDAE(input_dimensions=self.input_dimensions, num_layers=self.num_layers, latent_dimensions=self.latent_dimensions)
        if self.io_dict['in']['input_model_pth_path']:
            self.model.load_state_dict(torch.load(self.io_dict['in']['input_model_pth_path']))

        # Define loss function and optimizer algorithm
        try:
            self.loss_function: torch.nn.modules.loss._Loss = get_loss_function(properties.get('loss_function', ''))()
            fu.log(f'Using loss function: {self.loss_function}', self.out_log)
        except ValueError:
            fu.log(f'Invalid loss function: {self.loss_function}', self.out_log)
            fu.log('Using default loss function: MSELoss', self.out_log)
            self.loss_function = torch.nn.MSELoss()

        try:
            self.optimizer: torch.optim.Optimizer = get_optimizer_function(properties.get('optimizer', ''))(self.model.parameters(), lr=self.lr)
            fu.log(f'Using optimizer: {self.optimizer}', self.out_log)
        except ValueError:
            fu.log(f'Invalid optimizer: {self.optimizer}', self.out_log)
            fu.log('Using default optimizer: Adam', self.out_log)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @launchlogger
    def launch(self) -> int:
        """Execute the :class:`TrainMDAE <mdae.train_mdae.TrainMDAE>` object."""

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        # Train the model
        train_losses, validation_losses = self.train_model()
        if self.stage_io_dict['out']['output_train_data_npy_path']:
            np.save(self.stage_io_dict['out']['output_train_data_npy_path'], (np.array(train_losses), np.array(validation_losses)))

        # Evaluate the model
        if self.stage_io_dict['out']['output_performance_npy_path']:
            evaluate_losses, latent_space, reconstructed_data = self.evaluate_model(self.performance_dataloader, self.loss_function)
            denormalized_reconstructed_data = ndarray_denormalization(reconstructed_data, self.input_train_data_max_values, self.input_train_data_min_values)
            self.model.save_model(self.stage_io_dict['out']['output_performance_npy_path'], (evaluate_losses, latent_space, denormalized_reconstructed_data))

        # Save the model
        torch.save(self.model.state_dict(), self.stage_io_dict['out']['output_model_pth_path'])

        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)
        return 0

    def train_model(self) -> Tuple[List[float], List[float]]:
        self.model.to(self.model.device)
        train_losses: List[float] = []
        validation_losses: List[float] = []

        start_time: float = time.time()

        fu.log("Start Training:", self.out_log)
        for epoch_index in range(self.num_epochs):
            # Training step
            avg_train_loss: float = self.training_step(self.train_dataloader, self.optimizer, self.loss_function)
            train_losses.append(avg_train_loss)

            # Validation step
            avg_validation_loss, _, _ = self.evaluate_model(self.validation_dataloader, self.loss_function)
            validation_losses.append(avg_validation_loss)

            epoch_duration = time.time() - start_time

            # Logging
            if self.log_interval and (epoch_index % self.log_interval == 0 or epoch_index == self.num_epochs-1):
                fu.log(f'{"Epoch":>4} {epoch_index+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.3f}, Validation Loss: {avg_validation_loss:.3f}, Duration: {epoch_duration:.2f}s')
                start_time = time.time()

            # Save checkpoint
            if self.checkpoint_interval and (epoch_index % self.checkpoint_interval == 0 or epoch_index == self.num_epochs-1):
                checkpoint_path = str(Path(self.stage_io_dict.get("unique_dir", '')).joinpath(f'{self.output_checkpoint_prefix}_{epoch_index}.pth'))
                fu.log(f'{"Saving: ":>4} {checkpoint_path}', self.out_log)
                torch.save(self.model.state_dict(), checkpoint_path)

        return train_losses, validation_losses

    def training_step(self, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_function: torch.nn.modules.loss._Loss) -> float:
        self.model.train()
        losses: List[float] = []
        for data in dataloader:
            data = data.to(self.model.device)
            _, output = self.model(data)
            loss = loss_function(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses))

    def evaluate_model(self, dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.modules.loss._Loss) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        losses: List[float] = []
        z_list: List[float] = []
        x_hat_list: List[float] = []
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.model.device)
                latent, output = self.model(data)
                loss = loss_function(output, data)
                losses.append(loss.item())
                z_list.append(latent.cpu().numpy())
                x_hat_list.append(output.cpu().numpy())
        loss = float(np.mean(losses))
        latent_space: np.ndarray = np.reshape(np.concatenate(z_list, axis=0), (-1, self.latent_dimensions))
        reconstructed_data: np.ndarray = np.reshape(np.concatenate(x_hat_list, axis=0), (-1, int(self.input_dimensions)))
        return loss, latent_space, reconstructed_data


def trainMDAE(input_train_npy_path: str, output_model_pth_path: str, input_model_pth_path: Optional[str],
              output_train_data_npy_path: Optional[str], output_performance_npy_path: Optional[str],
              properties: Optional[Dict] = None, **kwargs) -> int:
    """Execute the :class:`TrainMDAE <mdae.train_mdae.TrainMDAE>` class and
    execute the :meth:`launch() <mdae.train_mdae.TrainMDAE.launch>` method."""

    return TrainMDAE(input_train_npy_path=input_train_npy_path, output_model_pth_path=output_model_pth_path, input_model_pth_path=input_model_pth_path,
                     output_train_data_npy_path=output_train_data_npy_path, output_performance_npy_path=output_performance_npy_path,
                     properties=properties, **kwargs).launch()


def main():
    """Command line execution of this building block. Please check the command line documentation."""
    parser = argparse.ArgumentParser(description='Train a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.',
                                     formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999))
    parser.add_argument('-c', '--config', required=False, help="This file can be a YAML file, JSON file or JSON string")

    # Specific args of each building block
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--input_train_npy_path', required=True, help='Path to the input train data file. Accepted formats: npy.')
    required_args.add_argument('--output_model_pth_path', required=True, help='Path to the output model file. Accepted formats: pth.')
    parser.add_argument('--input_model_pth_path', required=False, help='Path to the input model file. Accepted formats: pth.')
    parser.add_argument('--output_train_data_npy_path', required=False, help='Path to the output train data file. Accepted formats: npy.')
    parser.add_argument('--output_performance_npy_path', required=False, help='Path to the output performance file. Accepted formats: npy.')
    parser.add_argument('--properties', required=False, help='Additional properties for the MDAE object.')
    args = parser.parse_args()
    config = args.config if args.config else None
    properties = settings.ConfReader(config=config).get_prop_dic()

    trainMDAE(input_train_npy_path=args.input_train_npy_path, output_model_pth_path=args.output_model_pth_path,
              input_model_pth_path=args.input_model_pth_path, output_train_data_npy_path=args.output_train_data_npy_path,
              output_performance_npy_path=args.output_performance_npy_path, properties=properties)


if __name__ == '__main__':
    main()
