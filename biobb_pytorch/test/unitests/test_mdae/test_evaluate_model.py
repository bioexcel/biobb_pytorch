# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.evaluate_model import evaluate_model
import numpy as np


class TestEvaluateModel:
    def setup_class(self):
        fx.test_setup(self, 'evaluate_model')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_evaluate_model(self):
        evaluate_model(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_npz_path'])

        results = np.load(self.paths['output_results_npz_path'], allow_pickle=True)
        assert 'xhat' in results or 'z' in results or 'eval_loss' in results, "Results should contain evaluation data (xhat, z, or eval_loss)"
        
        if 'ref_output_results_npz_path' in self.paths:
            ref_results = np.load(self.paths['ref_output_results_npz_path'], allow_pickle=True)
            # Compare key metrics
            for key in ['loss', 'mse', 'reconstruction_error']:
                if key in results and key in ref_results:
                    assert isinstance(results[key], (np.ndarray, float)), f"{key} should be numeric"

