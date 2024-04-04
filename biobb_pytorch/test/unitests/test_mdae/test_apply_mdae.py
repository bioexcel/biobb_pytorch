from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.apply_mdae import applyMDAE


class TestTrainMdae:
    def setup_class(self):
        fx.test_setup(self, 'apply_mdae')

    def teardown_class(self):
        # pass
        fx.test_teardown(self)

    def test_mdae(self):
        applyMDAE(properties=self.properties, **self.paths)  # type: ignore
        assert fx.not_empty(self.paths['output_reconstructed_data_npy_path'])  # type: ignore
        assert fx.not_empty(self.paths['output_latent_space_npy_path'])  # type: ignore
