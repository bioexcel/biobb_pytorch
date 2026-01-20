# type: ignore
import pytest
import numpy as np
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from biobb_pytorch.mdae.plots import (
    plot_loss,
    plot_rmsd,
    plot_rmsf,
    plot_rmsf_difference,
    plot_latent_space
)


class TestPlots:
    def teardown_method(self):
        """Close all matplotlib figures after each test."""
        plt.close('all')

    def test_plot_loss_with_validation(self):
        """Test plot_loss with training and validation losses."""
        # Create temporary npz file with training and validation losses
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
            train_loss = np.array([1.0, 0.8, 0.6, 0.5, 0.45])
            valid_loss = np.array([1.1, 0.9, 0.65, 0.55, 0.5])
            np.savez(tmp_path, train_loss=train_loss, valid_loss=valid_loss)
        
        try:
            # This should not raise an error
            plot_loss(tmp_path)
            assert True
        finally:
            Path(tmp_path).unlink()

    def test_plot_loss_without_validation(self):
        """Test plot_loss with only training loss."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
            train_loss = np.array([1.0, 0.8, 0.6, 0.5, 0.45])
            np.savez(tmp_path, train_loss=train_loss)
        
        try:
            plot_loss(tmp_path)
            assert True
        finally:
            Path(tmp_path).unlink()

    def test_plot_rmsd_single_file(self):
        """Test plot_rmsd with a single XVG file."""
        with tempfile.NamedTemporaryFile(suffix='.xvg', delete=False, mode='w') as tmp:
            tmp_path = tmp.name
            # Write sample XVG data
            tmp.write("# Comment line\n")
            tmp.write("@ xaxis label \"Time (ns)\"\n")
            tmp.write("0.0 0.1\n")
            tmp.write("1.0 0.15\n")
            tmp.write("2.0 0.12\n")
        
        try:
            plot_rmsd(tmp_path)
            assert True
        finally:
            Path(tmp_path).unlink()

    def test_plot_rmsd_multiple_files(self):
        """Test plot_rmsd with multiple XVG files."""
        tmp_paths = []
        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(suffix=f'_{i}.xvg', delete=False, mode='w') as tmp:
                    tmp_path = tmp.name
                    tmp_paths.append(tmp_path)
                    tmp.write("# Comment line\n")
                    tmp.write("0.0 0.1\n")
                    tmp.write("1.0 0.15\n")
            
            plot_rmsd(tmp_paths)
            assert True
        finally:
            for path in tmp_paths:
                Path(path).unlink()

    def test_plot_rmsd_nonexistent_file(self, capsys):
        """Test plot_rmsd with non-existent file."""
        plot_rmsd("/nonexistent/file.xvg")
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "does not exist" in captured.out

    def test_plot_rmsf_single_file(self):
        """Test plot_rmsf with a single XVG file."""
        with tempfile.NamedTemporaryFile(suffix='.xvg', delete=False, mode='w') as tmp:
            tmp_path = tmp.name
            tmp.write("# Comment line\n")
            tmp.write("@ xaxis label \"Residue\"\n")
            tmp.write("1 0.1\n")
            tmp.write("2 0.15\n")
            tmp.write("3 0.12\n")
        
        try:
            plot_rmsf(tmp_path)
            assert True
        finally:
            Path(tmp_path).unlink()

    def test_plot_rmsf_multiple_files(self):
        """Test plot_rmsf with multiple XVG files."""
        tmp_paths = []
        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(suffix=f'_{i}.xvg', delete=False, mode='w') as tmp:
                    tmp_path = tmp.name
                    tmp_paths.append(tmp_path)
                    tmp.write("# Comment line\n")
                    tmp.write("1 0.1\n")
                    tmp.write("2 0.15\n")
            
            plot_rmsf(tmp_paths)
            assert True
        finally:
            for path in tmp_paths:
                Path(path).unlink()

    def test_plot_rmsf_difference(self):
        """Test plot_rmsf_difference with two XVG files."""
        tmp_paths = []
        try:
            for i, values in enumerate([[0.1, 0.15, 0.12], [0.12, 0.14, 0.11]]):
                with tempfile.NamedTemporaryFile(suffix=f'_{i}.xvg', delete=False, mode='w') as tmp:
                    tmp_path = tmp.name
                    tmp_paths.append(tmp_path)
                    tmp.write("# Comment line\n")
                    for res, val in enumerate(values, 1):
                        tmp.write(f"{res} {val}\n")
            
            plot_rmsf_difference(tmp_paths)
            assert True
        finally:
            for path in tmp_paths:
                Path(path).unlink()

    def test_plot_latent_space_basic(self):
        """Test plot_latent_space with basic data."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
            # Create latent space data
            z = np.random.randn(100, 5)  # 100 frames, 5 dimensions
            np.savez(tmp_path, z=z)
        
        try:
            plot_latent_space(tmp_path, projection_dim=[0, 1])
            assert True
        finally:
            Path(tmp_path).unlink()

    def test_plot_latent_space_custom_dims(self):
        """Test plot_latent_space with custom projection dimensions."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
            z = np.random.randn(100, 5)
            np.savez(tmp_path, z=z)
        
        try:
            plot_latent_space(tmp_path, projection_dim=[2, 3], snapshot_freq_ps=20)
            assert True
        finally:
            Path(tmp_path).unlink()

    def test_plot_latent_space_missing_z(self):
        """Test plot_latent_space raises KeyError when 'z' is missing."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
            # Save data without 'z' key
            np.savez(tmp_path, other_data=np.array([1, 2, 3]))
        
        try:
            with pytest.raises(KeyError, match="'z' not found"):
                plot_latent_space(tmp_path, projection_dim=[0, 1])
        finally:
            Path(tmp_path).unlink()

    def test_plot_latent_space_invalid_projection_dim(self):
        """Test plot_latent_space raises ValueError for invalid projection_dim."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
            z = np.random.randn(100, 5)
            np.savez(tmp_path, z=z)
        
        try:
            with pytest.raises(ValueError, match="projection_dim must have length 2"):
                plot_latent_space(tmp_path, projection_dim=[0, 1, 2])
        finally:
            Path(tmp_path).unlink()

