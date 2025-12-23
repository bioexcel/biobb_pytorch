# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.mdfeaturizer import mdfeaturizer


class TestMDFeaturizer:
    def setup_class(self):
        fx.test_setup(self, 'mdfeaturizer')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_mdfeaturizer(self):
        mdfeaturizer(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_dataset_pt_path'])
        assert fx.not_empty(self.paths['output_stats_pt_path'])
        assert fx.equal(self.paths['output_dataset_pt_path'], self.paths['ref_output_dataset_pt_path'])
        assert fx.equal(self.paths['output_stats_pt_path'], self.paths['ref_output_stats_pt_path'])
