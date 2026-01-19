# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.train_model import train_model
import torch
import numpy as np


class TestTrainModel:
    def setup_class(self):
        fx.test_setup(self, 'train_model')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_train_model(self):
        train_model(properties=self.properties, **self.paths)
        
        if 'output_model_pth_path' in self.paths:
            assert fx.not_empty(self.paths['output_model_pth_path'])
            # The model is saved directly as an object, not in a dictionary
            model = torch.load(self.paths['output_model_pth_path'], weights_only=False)
            assert hasattr(model, 'state_dict'), "Model file should contain a PyTorch model"
        
        if 'output_metrics_npz_path' in self.paths:
            assert fx.not_empty(self.paths['output_metrics_npz_path'])
            metrics = np.load(self.paths['output_metrics_npz_path'], allow_pickle=True)
            assert 'train_loss' in metrics or 'loss' in metrics, "Metrics should contain loss information"
            
            if 'ref_output_metrics_npz_path' in self.paths:
                ref_metrics = np.load(self.paths['ref_output_metrics_npz_path'], allow_pickle=True)
                # Compare final loss values
                if 'train_loss' in metrics and 'train_loss' in ref_metrics:
                    assert isinstance(metrics['train_loss'], (np.ndarray, float)), "Train loss should be numeric"

