# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.explainability.LRP import relevance_propagation
import numpy as np


class TestLRP:
    def setup_class(self):
        fx.test_setup(self, 'lrp')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_lrp(self):
        relevance_propagation(properties=self.properties, **self.paths)
        
        if 'output_results_npz_path' in self.paths and self.paths['output_results_npz_path']:
            assert fx.not_empty(self.paths['output_results_npz_path'])
            
            results = np.load(self.paths['output_results_npz_path'], allow_pickle=True)
            assert 'global_importance' in results, "Results should contain global_importance scores"
            
            if 'ref_output_results_npz_path' in self.paths:
                ref_results = np.load(self.paths['ref_output_results_npz_path'], allow_pickle=True)
                # Compare relevance scores structure
                if 'global_importance' in results and 'global_importance' in ref_results:
                    assert results['global_importance'].shape == ref_results['global_importance'].shape, "Global importance scores should have matching shapes"

