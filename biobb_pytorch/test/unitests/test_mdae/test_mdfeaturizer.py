# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.mdfeaturizer import mdfeaturizer
import torch


class TestMDFeaturizer:
    def setup_class(self):
        fx.test_setup(self, 'mdfeaturizer')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_mdfeaturizer(self):
        mdfeaturizer(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_dataset_pt_path'])
        assert fx.not_empty(self.paths['output_stats_pt_path'])

        dataset = torch.load(self.paths['output_dataset_pt_path'], weights_only=False)
        ref_dataset = torch.load(self.paths['ref_output_dataset_pt_path'], weights_only=False)
        assert (dataset['data'] == ref_dataset['data']).all(), "Datasets are not equal!"

        stats = torch.load(self.paths['output_stats_pt_path'], weights_only=False)
        ref_stats = torch.load(self.paths['ref_output_stats_pt_path'], weights_only=False)
        for key in stats.keys():
            if key == 'topology':
                continue
            if isinstance(stats[key], torch.Tensor):
                assert (stats[key] == ref_stats[key]).all(), f"Stats for {key} are not equal!"
            else:
                assert stats[key] == ref_stats[key], f"Stats for {key} are not equal!"
