# type: ignore
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.feat2traj import feat2traj
import os


class TestFeat2Traj:
    def setup_class(self):
        fx.test_setup(self, 'feat2traj')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_feat2traj(self):
        feat2traj(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_traj_path'])
        
        # Verify trajectory file exists and has reasonable size
        assert os.path.exists(self.paths['output_traj_path']), "Trajectory file should exist"
        file_size = os.path.getsize(self.paths['output_traj_path'])
        assert file_size > 0, "Trajectory file should not be empty"
        
        if 'output_top_path' in self.paths and self.paths['output_top_path']:
            assert fx.not_empty(self.paths['output_top_path'])
            assert os.path.exists(self.paths['output_top_path']), "Topology file should exist"
        
        if 'ref_output_traj_path' in self.paths:
            ref_size = os.path.getsize(self.paths['ref_output_traj_path'])
            # Compare file sizes (should be similar for same input)
            assert abs(file_size - ref_size) / max(file_size, ref_size) < 0.1, "Trajectory sizes should be similar"

