from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.train_mdae import trainMDAE


class TestTrainMdae:
    def setup_class(self):
        fx.test_setup(self, 'train_mdae')

    def teardown_class(self):
        pass
        # fx.test_teardown(self)

    def test_mdae(self):
        trainMDAE(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_pth_path'])
        assert fx.equal(self.paths['output_model_pth_path'], self.paths['ref_output_model_pth_path'])
