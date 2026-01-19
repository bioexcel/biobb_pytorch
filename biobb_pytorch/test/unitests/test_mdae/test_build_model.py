# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.build_model import build_model
import torch


class TestBuildModel:
    def setup_class(self):
        fx.test_setup(self, 'build_model')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_build_model(self):
        build_model(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_pth_path'])

        # Load and verify model structure
        # The model is saved directly as an object, not in a dictionary
        model = torch.load(self.paths['output_model_pth_path'], weights_only=False)
        # Verify it's a PyTorch model (has state_dict method)
        assert hasattr(model, 'state_dict'), "Model file should contain a PyTorch model"
        assert hasattr(model, 'encoder') or hasattr(model, '_hparams'), "Model should have encoder or _hparams attribute"

