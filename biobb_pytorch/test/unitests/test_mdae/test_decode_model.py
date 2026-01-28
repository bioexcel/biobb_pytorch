# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.decode_model import evaluateDecoder
import numpy as np


class TestDecodeModel:
    def setup_class(self):
        fx.test_setup(self, 'evaluateDecoder')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_decode_model(self):
        evaluateDecoder(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_results_npz_path'])

        results = np.load(self.paths['output_results_npz_path'], allow_pickle=True)
        assert 'xhat' in results, "Results should contain decoded/reconstructed data (xhat)"
        
        if 'ref_output_results_npz_path' in self.paths:
            ref_results = np.load(self.paths['ref_output_results_npz_path'], allow_pickle=True)
            # Compare decoded data dimensions
            if 'xhat' in results and 'xhat' in ref_results:
                assert results['xhat'].shape[0] == ref_results['xhat'].shape[0], "Number of samples should match"

