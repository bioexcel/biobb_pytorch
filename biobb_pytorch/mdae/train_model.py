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

class TrainModel(BiobbObject):
    """
    | biobb_pytorch TrainModel
    | Trains a PyTorch autoencoder using the given properties.
    | Trains a PyTorch autoencoder using the given properties.

    Args:
        input_model_pth_path (str): Path to the input model file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        input_dataset_pt_path (str): Path to the input dataset file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pt>`_. Accepted formats: pt (edam:format_2333).
        output_model_pth_path (str) (Optional): Path to the output model file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        output_metrics_npz_path (str) (Optional): Path to the output metrics file. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.npz>`_. Accepted formats: npz (edam:format_2333).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **Trainer** (*dict*) - ({}) Trainer options from PyTorch Lightning.
            * **Dataset** (*dict*) - ({}) Dataset options from mlcolvar.

    Examples:
        This example shows how to use the TrainModel class to train a PyTorch autoencoder model::

            from biobb_pytorch.mdae.train_model import TrainModel
            
            input_model_pth_path='input_model.pth'
            input_dataset_pt_path='input_dataset.pt'
            output_model_pth_path='output_model.pth'
            output_metrics_npz_path='output_metrics.npz'

            prop={
                'Trainer': {
                    'max_epochs': 10,
                    'callbacks': {
                        'metrics': ['EarlyStopping']
                        }
                    }
                },
                'Dataset': {
                    'batch_size': 32,
                    'split': {
                        'train_prop': 0.8,
                        'val_prop': 0.2
                    }
                }
            }
            
            # For API usage, outputs can be None to avoid saving
            instance = TrainModel(input_model_pth_path=input_model_pth_path,
                                  input_dataset_pt_path=input_dataset_pt_path,
                                  output_model_pth_path=None,
                                  output_metrics_npz_path=None,
                                  properties=prop)
            instance.launch()
            trained_model = instance.model  # Access the trained PyTorch model
            metrics = instance.metrics  # Access the training metrics

            # Or to save, provide outputs and call launch
            instance = TrainModel(input_model_pth_path=input_model_pth_path,
                                  input_dataset_pt_path=input_dataset_pt_path,
                                  output_model_pth_path=output_model_pth_path,
                                  output_metrics_npz_path=output_metrics_npz_path,
                                  properties=prop)
            instance.launch()
    
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
        output_model_pth_path: Optional[str] = None,
        output_metrics_npz_path: Optional[str] = None,
        properties: dict = None,
    )-> None:
        
        properties = properties or {}

        super().__init__(properties)

        self.input_model_pth_path = input_model_pth_path
        self.input_dataset_pt_path = input_dataset_pt_path
        self.output_model_pth_path = output_model_pth_path
        self.output_metrics_npz_path = output_metrics_npz_path
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
        if output_model_pth_path:
            self.io_dict["out"]["output_model_pth_path"] = output_model_pth_path
        if output_metrics_npz_path:
            self.io_dict["out"]["output_metrics_npz_path"] = output_metrics_npz_path

        self.Trainer = self.properties.get('Trainer', {})
        self.Dataset = self.properties.get('Dataset', {})

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

    def get_callbacks(self):
        self.colvars_metrics = MetricsCallback()
        cbs_list = [self.colvars_metrics]

        callbacks_prop = self.properties.get('Trainer', {}).get('callbacks', {})
        if not callbacks_prop:
            return cbs_list
        else:
            for k, v in self.properties['Trainer']['callbacks'].items():
                callback_params = self.properties['Trainer']['callbacks'][k]
                CallbackClass = getattr(_cbs, k, None)
                if CallbackClass:
                    callback = CallbackClass(**callback_params)
                    cbs_list.append(callback)
            return cbs_list

    def get_logger(self):
        logger_prop = self.properties.get('Trainer', {}).get('logger', False)
        if not logger_prop:
            return None

        logger_type, logger_params = next(iter(logger_prop.items()))
        LoggerClass = getattr(_loggers, logger_type, None)
        if LoggerClass is None:
            raise KeyError(f"No Logger named {logger_type} in lightning.pytorch.loggers")

        return LoggerClass(**logger_params)

    def get_profiler(self):
        profiler_prop = self.properties.get('Trainer', {}).get('profiler')
        if not profiler_prop:
            return None

        profiler_type, profiler_params = next(iter(profiler_prop.items()))
        ProfilerClass = getattr(_profiler, profiler_type, None)
        if ProfilerClass is None:
            raise KeyError(f"No Profiler named {profiler_type} in lightning.pytorch.profilers")

        return ProfilerClass(**profiler_params)

    def get_trainer(self):
        train_params = {k: v for k, v in self.properties['Trainer'].items()
                        if k not in ['callbacks', 'logger', 'profiler']}
        train_params['callbacks'] = self.get_callbacks()
        train_params['logger'] = self.get_logger()
        train_params['profiler'] = self.get_profiler()
        return lightning.Trainer(**train_params)

    def load_model(self):
        return torch.load(self.io_dict["in"]["input_model_pth_path"], 
                          weights_only=False)

    def load_dataset(self):
        dataset = torch.load(self.io_dict["in"]["input_dataset_pt_path"], 
                             weights_only=False)
        return DictDataset(dataset)

    def create_datamodule(self, dataset):
        ds_cfg = self.properties['Dataset']

        lengths = [ds_cfg['split'].get('train_prop', 0.8),
                   ds_cfg['split'].get('val_prop', 0.2)]
        if ds_cfg['split'].get('test_prop', 0) > 0:
            lengths.append(ds_cfg['split'].get('test_prop', 0))

        return DictModule(
            dataset,
            batch_size=ds_cfg.get('batch_size', 16),
            lengths=lengths,
            shuffle=ds_cfg['split'].get('shuffle', True),
            random_split=ds_cfg['split'].get('random_split', True)
        )

    def fit_model(self, trainer, model, datamodule):
        """Fit the model to the data, capturing logs and keeping tqdm clean."""
        trainer.fit(model, datamodule)

    def save_full(self, model) -> None:
        """Serialize the full model object (including architecture)."""
        torch.save(model, self.io_dict["out"]["output_model_pth_path"])

    @launchlogger
    def launch(self) -> int:
        """
        Execute the :class:`TrainModel` class and its `.launch()` method.
        """

        fu.log(f'## BioBB Model Trainer ##', self.out_log)

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        # Start Pipeline

        # load the model
        fu.log(f'Load model from {os.path.abspath(self.io_dict["in"]["input_model_pth_path"])}', self.out_log)
        self.model = self.load_model()

        # load the dataset
        fu.log(f'Load dataset from {os.path.abspath(self.io_dict["in"]["input_dataset_pt_path"])}', self.out_log)
        self.dataset = self.load_dataset()
        
        # create the datamodule
        fu.log(f'Start training...', self.out_log)
        self.datamodule = self.create_datamodule(self.dataset)

        # get the trainer
        self.trainer = self.get_trainer()

        # fit the model
        self.fit_model(self.trainer, self.model, self.datamodule)
        
        # Set the metrics
        self.metrics = self.colvars_metrics.metrics

        # Save the metrics if path provided
        if self.output_metrics_npz_path:
            np.savez_compressed(self.io_dict["out"]["output_metrics_npz_path"], **self.metrics)
            fu.log(f'Training Metrics saved to {os.path.abspath(self.io_dict["out"]["output_metrics_npz_path"])}', self.out_log)
            fu.log(f'File size: {get_size(self.io_dict["out"]["output_metrics_npz_path"])}', self.out_log)

        # save the model if path provided
        if self.output_model_pth_path:
            self.save_full(self.model)
            fu.log(f'Trained Model saved to {os.path.abspath(self.io_dict["out"]["output_model_pth_path"])}', self.out_log)
            fu.log(f'File size: {get_size(self.io_dict["out"]["output_model_pth_path"])}', self.out_log)

        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        output_created = bool(self.output_model_pth_path or self.output_metrics_npz_path)
        self.check_arguments(output_files_created=output_created, raise_exception=False)

        return 0 
        
def trainModel(
    properties: dict,
    input_model_pth_path: str,
    input_dataset_pt_path: str,
    output_model_pth_path: Optional[str] = None,
    output_metrics_npz_path: Optional[str] = None,
) -> int:
    """
    Execute the :class:`TrainModel <TrainModel.TrainModel>` class and
    execute the :meth:`launch() <TrainModel.trainModel.launch>` method.
    """
    return TrainModel(
        input_model_pth_path=input_model_pth_path,
        input_dataset_pt_path=input_dataset_pt_path,
        output_model_pth_path=output_model_pth_path,
        output_metrics_npz_path=output_metrics_npz_path,
        properties=properties,
    ).launch()

trainModel.__doc__ = TrainModel.__doc__

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
        "-o",
        "--output_model_pth_path",
        required=False,
        help="Trajectory file path"
    )

    required_args.add_argument(
        "-om",
        "--output_metrics_npz_path",
        required=False,
        help="Trajectory file path"
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

    trainModel(
        input_model_pth_path=args.input_model_pth_path,
        input_dataset_pt_path=args.input_dataset_pt_path,
        output_model_pth_path=args.output_model_pth_path,
        output_metrics_npz_path=args.output_metrics_npz_path,
        properties=properties,
    )

if __name__ == "__main__":
    main()