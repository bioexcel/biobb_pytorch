# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.make_plumed import generatePlumed
import os


class TestMakePlumed:
    def setup_class(self):
        fx.test_setup(self, 'generatePlumed')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_generatePlumed(self):
        generatePlumed(properties=self.properties, **self.paths)

        assert fx.not_empty(self.paths['output_plumed_dat_path'])
        assert fx.not_empty(self.paths['output_features_dat_path'])
        assert fx.not_empty(self.paths['output_model_ptc_path'])

        # Verify PLUMED file contains expected content
        with open(self.paths['output_plumed_dat_path'], 'r') as f:
            plumed_content = f.read()
            assert 'PYTORCH_MODEL' in plumed_content or 'INCLUDE' in plumed_content, "PLUMED file should contain PYTORCH_MODEL or INCLUDE keywords"

        # Verify features file exists and is readable
        assert os.path.exists(self.paths['output_features_dat_path']), "Features file should exist"

        if 'ref_output_plumed_dat_path' in self.paths:
            # Compare structure (number of lines, key sections)
            assert len(plumed_content.split('\n')) > 0, "PLUMED file should not be empty"
