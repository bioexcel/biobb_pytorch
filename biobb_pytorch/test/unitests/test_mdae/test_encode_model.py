# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.encode_model import evaluate_encoder
import numpy as np


class TestEncodeModel:
    def setup_class(self):
        fx.test_setup(self, 'encode_model')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_encode_model(self):
        evaluate_encoder(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_npz_path'])

        results = np.load(self.paths['output_results_npz_path'], allow_pickle=True)
        assert 'latent' in results or 'encoded' in results or 'z' in results, "Results should contain encoded/latent data"
        
        if 'ref_output_results_npz_path' in self.paths:
            ref_results = np.load(self.paths['ref_output_results_npz_path'], allow_pickle=True)
            # Compare latent space dimensions
            if 'latent' in results and 'latent' in ref_results:
                assert results['latent'].shape[0] == ref_results['latent'].shape[0], "Number of samples should match"

