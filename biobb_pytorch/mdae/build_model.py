import os
import torch
import importlib
from typing import Dict, Any, Type, Optional
from biobb_pytorch.mdae.models import __all__ as AVAILABLE_MODELS
from biobb_pytorch.mdae.utils.model_utils import assert_valid_kwargs
from biobb_pytorch.mdae.utils.log_utils import get_size
from biobb_common.tools.file_utils import launchlogger
from biobb_common.tools import file_utils as fu
from biobb_common.generic.biobb_object import BiobbObject


class BuildModel(BiobbObject):
    """
    | biobb_pytorch BuildModel
    | Build a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
    | Builds a PyTorch autoencoder from the given properties.

    Args:
        input_stats_pt_path (str): Path to the input model statistics file. File type: input. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_model.pt>`_. Accepted formats: pt (edam:format_2333).
        output_model_pth_path (str) (Optional): Path to save the model in .pth format. File type: output. `Sample file <https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth>`_. Accepted formats: pth (edam:format_2333).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **model_type** (*str*) - ("AutoEncoder") Name of the model class to instantiate (must exist in biobb_pytorch.mdae.models).
            * **n_cvs** (*int*) - (1) Dimensionality of the latent space.
            * **encoder_layers** (*list*) - ([16]) List of integers representing the number of neurons in each encoder layer.
            * **decoder_layers** (*list*) - ([16]) List of integers representing the number of neurons in each decoder layer.
            * **options** (*dict*) - ({"norm_in": {"mode": "min_max"}}) Additional options (e.g. norm_in, optimizer, loss_function, device, etc.).

    Examples:
        This example shows how to use the BuildModel class to build a PyTorch autoencoder model::

            from biobb_pytorch.mdae.build_model import buildModel

            input_stats_pt_path = "input_stats.pt"
            output_model_pth_file = "model.pth"

            n_features = 128
            prop = {
                'model_type': 'AutoEncoder',
                'n_cvs': 10,
                'encoder_layers': [n_features, 64, 32],
                'decoder_layers': [32, 64, n_features],
                'options': {
                    'norm_in': {"mode": "min_max"},
                    'optimizer': {
                        'lr': 1e-4
                    }
                }
            }

            buildModel(input_stats_pt_path=input_stats_pt_path,
                       output_model_pth_path=None,
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
        input_stats_pt_path: str,
        output_model_pth_path: Optional[str] = None,
        properties: dict = None,
        **kwargs,
    ) -> None:

        properties = properties or {}

        super().__init__(properties)

        self.input_stats_pt_path = input_stats_pt_path
        self.output_model_pth_path = output_model_pth_path
        self.props = properties.copy()
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = {
            "in": {
                "input_stats_pt_path": input_stats_pt_path,
            },
            "out": {}
        }
        if output_model_pth_path:
            self.io_dict["out"]["output_model_pth_path"] = output_model_pth_path

        # build the per-feature arguments
        self.options: dict = properties.get("options", {})
        self.model_type: str = properties.get("model_type", "AutoEncoder")
        self.n_cvs: int = properties.get("n_cvs", 1)
        self.encoder_layers: list = properties.get("encoder_layers", [16])
        self.decoder_layers: list = properties.get("decoder_layers", [16])
        self.loss_function: Optional[dict] = properties.get("loss_function", None)
        self.device = self.options['device'] if 'device' in self.options else 'cpu'

        # load the input files
        self.stats = torch.load(self.io_dict['in']['input_stats_pt_path'],
                                weights_only=False)

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

        self._validate_props()
        self.model = self._build_model()
        self.loss_fn = self._build_loss()

        # Store hyperparameters for reproducibility
        hparams = {
            'model_type': properties['model_type'],
            'n_cvs': properties['n_cvs'],
            'encoder_layers': properties['encoder_layers'],
            'decoder_layers': properties['decoder_layers'],
            'loss_function': self._hparams_loss_repr(),
            'options': {k: v for k, v in properties['options'].items() if k != 'loss_function'}
        }
        setattr(self.model, '_hparams', hparams)

        # Attach loss_fn and move model to device
        self.model.loss_fn = self.loss_fn
        self.model.to(self.device)

    def _validate_props(self) -> None:
        required = ['model_type', 'n_cvs', 'encoder_layers', 'decoder_layers', 'options']
        missing = [k for k in required if k not in self.props]
        if missing:
            raise KeyError(f"Missing required properties: {missing}")

        model_type = self.props['model_type']
        if model_type not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Available: {AVAILABLE_MODELS}"
            )

    def _build_model(self) -> torch.nn.Module:
        module = importlib.import_module('biobb_pytorch.mdae.models')
        ModelClass: Type[torch.nn.Module] = getattr(module, self.props['model_type'])

        init_args = {
            'n_features': self.stats['shape'][1],
            'n_cvs': self.props['n_cvs'],
            'encoder_layers': self.props['encoder_layers'],
            'decoder_layers': self.props['decoder_layers'],
            'options': {k: v for k, v in self.props['options'].items() if k not in ['loss_function', 'norm_in']}
        }

        if 'norm_in' in self.props.get('options', {}):

            init_args['options']['norm_in'] = {
                'stats': self.stats,
                'mode': self.props['options']['norm_in'].get('mode')
            }

        assert_valid_kwargs(ModelClass, init_args, context="model init")

        return ModelClass(**init_args)

    def _build_loss(self) -> torch.nn.Module:
        loss_config = self.props['options'].get('loss_function')
        if loss_config and 'loss_type' in loss_config and loss_config['loss_type'] == 'PhysicsLoss':
            loss_config['stats'] = self.stats

        if not loss_config:
            # Use model's default
            return getattr(self.model, 'loss_fn', None)

        loss_type = loss_config.get('loss_type')
        if not loss_type:
            raise KeyError("'loss_type' must be specified in options['loss_function']")

        loss_module = importlib.import_module('biobb_pytorch.mdae.loss')
        LossClass = getattr(loss_module, loss_type)

        kwargs = {k: v for k, v in loss_config.items() if k != 'loss_type'}

        assert_valid_kwargs(LossClass, kwargs, context="loss init")
        try:
            return LossClass(**kwargs)
        except Exception:
            kwargs = {k: v for k, v in kwargs.items() if k != 'stats'}
            return LossClass(**kwargs)

    def _hparams_loss_repr(self) -> str:
        loss_config = self.props['options'].get('loss_function')
        if loss_config:
            name = loss_config.get('loss_type', '')
            args = [f"{k}={v}" for k, v in loss_config.items() if k not in ['loss_type', 'stats']]
            return f"{name}({', '.join(args)})"
        # fallback to model's representation
        return repr(getattr(self.model, 'loss_fn', ''))

    def save_weights(self, path: str) -> None:
        """Save model.state_dict() to the given path."""
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load_weights(
        cls,
        props: Dict[str, Any],
        path: str
    ) -> 'BuildModel':
        """Instantiate from props and load state_dict from path."""
        inst = cls(props)
        state = torch.load(path, map_location=inst.device)
        inst.model.load_state_dict(state)
        inst.model.to(inst.device)
        return inst

    def save_full(self) -> None:
        """Serialize the full model object (including architecture)."""
        torch.save(self.model, self.output_model_pth_path)

    @staticmethod
    def load_full(path: str) -> torch.nn.Module:
        """Load a model serialized with save_full."""
        return torch.load(path, weights_only=False)

    @launchlogger
    def launch(self) -> int:
        """
        Execute the :class:`BuildModel` class and its `.launch()` method.

        Args:
            output_model_pth_path (str): Path where the model will be saved.
            properties (dict): Hyper‐parameters for model construction.
        """

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        if self.output_model_pth_path:
            self.save_full()

        fu.log("## BioBB AutoEncoder Builder ##", self.out_log)
        fu.log("", self.out_log)
        fu.log("Hyperparameters:", self.out_log)
        fu.log("----------------", self.out_log)
        for key, value in self.model.__dict__.get('_hparams').items():
            if key == 'options':
                fu.log(f"{key}:", self.out_log)
                for sub_key, sub_value in value.items():
                    fu.log(f"   {sub_key}: {sub_value}", self.out_log)
            else:
                fu.log(f"{key}: {value}", self.out_log)
        fu.log("", self.out_log)

        fu.log("Model:", self.out_log)
        fu.log("------", self.out_log)

        for line in str(self.model).splitlines():
            fu.log(line, self.out_log)
        fu.log("", self.out_log)

        if self.output_model_pth_path:
            fu.log(f"Model saved in .pth format in "
                   f'{os.path.abspath(self.io_dict["out"]["output_model_pth_path"])}',
                   self.out_log,
                   )
            fu.log(f'File size: '
                   f'{get_size(self.io_dict["out"]["output_model_pth_path"])}',
                   self.out_log,
                   )

        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        self.check_arguments(output_files_created=bool(self.output_model_pth_path), raise_exception=False)

        return 0


def buildModel(
    properties: dict,
    input_stats_pt_path: str,
    output_model_pth_path: Optional[str] = None,
    **kwargs,
) -> int:
    """Create the :class:`BuildModel <BuildModel.BuildModel>` class and
    execute the :meth:`launch() <BuildModel.BuildModel.launch>` method."""
    return BuildModel(**dict(locals())).launch()


buildModel.__doc__ = BuildModel.__doc__
main = BuildModel.get_main(buildModel, "Build a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.")


if __name__ == "__main__":
    main()

# Example usage:

# n_features = torch.rand(100, 20)
# n_feat = n_features.shape[1]

# properties = {
#             'model_type': 'VariationalAutoEncoder',
#             'n_cvs': 10,
#             'encoder_layers': [n_feat, 64, 32],
#             'decoder_layers': [32, 64, n_feat],
#             'options': {
#                 'loss_function': {
#                     'loss_type': 'ELBOLoss',
#                     'beta': 1.0,
#                     'reconstruction': 'mse',
#                     'reduction': 'sum'},
#                 'optimizer': {
#                     'lr': 0.001
#                 }
#             }
#         }

# model_builder = BuildModel(properties)
# model_builder.save_full("test_model.pth")
# model = model_builder.load_full("test_model.pth")

# print()
# print("Hyperparameters:")
# print("----------------")
# for key, value in model._hparams.items():
#     print(f"{key}: {value}")
# print()
# print("Model:")
# print("------")
# print(model)
