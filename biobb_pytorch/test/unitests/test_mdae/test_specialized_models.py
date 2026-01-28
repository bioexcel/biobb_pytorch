# type: ignore
"""
Test suite for specialized model types that require specific configurations.

This includes:
- GaussianMixtureVariationalAutoEncoder (GMVAE): requires dict-based encoder/decoder config
- SPIB: requires k parameter and specific configuration
- CNNAutoEncoder (MoLearn): requires different architecture

These models cannot be tested with the standard list-based encoder/decoder layer
configuration and require specialized setups.
"""
import pytest
import torch
import tempfile
from pathlib import Path
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.build_model import buildModel, BuildModel


class TestGMVAE:
    """Test suite for GaussianMixtureVariationalAutoEncoder."""

    def setup_class(self):
        """Setup test fixtures."""
        fx.test_setup(self, 'buildModel')

    def teardown_class(self):
        """Cleanup after tests."""
        fx.test_teardown(self)

    def test_build_gmvae_with_proper_config(self):
        """Test building GMVAE with proper dictionary-based encoder/decoder configuration."""
        props = self.properties.copy()
        props['model_type'] = 'GaussianMixtureVariationalAutoEncoder'

        # GMVAE requires dictionary-based encoder/decoder layers with specific
        # keys
        props['encoder_layers'] = {
            'qy_dims': [32, 16],  # Cluster assignment network
            'qz_dims': [32, 16]   # Latent variable network
        }
        props['decoder_layers'] = {
            'pz_dims': [16, 32],  # Latent prior network
            'px_dims': [16, 32]   # Reconstruction network
        }

        # GMVAE also requires 'k' (number of clusters) in options
        props['options']['k'] = 3
        props['options']['encoder'] = {
            'qy_nn': {},
            'qz_nn': {}
        }
        props['options']['decoder'] = {
            'pz_nn': {},
            'px_nn': {}
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # This should work with proper configuration
            buildModel(
                properties=props,
                input_stats_pt_path=self.paths['input_stats_pt_path'],
                output_model_pth_path=tmp_path
            )

            assert Path(tmp_path).exists(
            ), "GMVAE model file should be created"

            model = torch.load(tmp_path, weights_only=False)
            assert model.__class__.__name__ == 'GaussianMixtureVariationalAutoEncoder'
            assert hasattr(model, 'encoder'), "GMVAE should have encoder"
            assert hasattr(model, 'decoder'), "GMVAE should have decoder"
            assert model.k == 3, "GMVAE should have k=3"

        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()


class TestSPIB:
    """Test suite for SPIB model."""

    def setup_class(self):
        """Setup test fixtures."""
        fx.test_setup(self, 'buildModel')

    def teardown_class(self):
        """Cleanup after tests."""
        fx.test_teardown(self)

    def test_build_spib_with_proper_config(self):
        """Test building SPIB with list-based configuration and k parameter."""
        props = self.properties.copy()
        props['model_type'] = 'SPIB'

        # SPIB uses list-based layers but requires 'k' in options
        props['encoder_layers'] = [32, 16]
        props['decoder_layers'] = [16, 32]
        props['options']['k'] = 2  # Number of states

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            buildModel(
                properties=props,
                input_stats_pt_path=self.paths['input_stats_pt_path'],
                output_model_pth_path=tmp_path
            )

            assert Path(tmp_path).exists(), "SPIB model file should be created"

            model = torch.load(tmp_path, weights_only=False)
            assert model.__class__.__name__ == 'SPIB'
            assert hasattr(model, 'encoder'), "SPIB should have encoder"
            assert hasattr(model, 'decoder'), "SPIB should have decoder"
            assert model.k == 2, "SPIB should have k=2"

        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_spib_forward_pass(self):
        """Test SPIB forward pass."""
        props = self.properties.copy()
        props['model_type'] = 'SPIB'
        props['encoder_layers'] = [24, 12]
        props['decoder_layers'] = [12, 24]
        props['options']['k'] = 2

        instance = BuildModel(
            input_stats_pt_path=self.paths['input_stats_pt_path'],
            output_model_pth_path=None,
            properties=props
        )

        model = instance.model
        stats = torch.load(
            self.paths['input_stats_pt_path'],
            weights_only=False)
        n_features = stats['shape'][1]

        batch_size = 4
        dummy_input = torch.randn(batch_size, n_features)

        model.eval()
        with torch.no_grad():
            try:
                output = model(dummy_input)
                # SPIB should return tensor or dict with latent representation
                assert output is not None, "SPIB should produce output"
            except Exception as e:
                pytest.fail(f"SPIB forward pass failed: {str(e)}")


class TestCNNAutoEncoder:
    """Test suite for CNNAutoEncoder (MoLearn) model."""

    def setup_class(self):
        """Setup test fixtures."""
        fx.test_setup(self, 'buildModel')

    def teardown_class(self):
        """Cleanup after tests."""
        fx.test_teardown(self)

    @pytest.mark.skip(reason="CNNAutoEncoder requires specialized 3D input configuration")
    def test_build_cnn_autoencoder(self):
        """
        Test building CNNAutoEncoder.

        Note: CNNAutoEncoder (from MoLearn) is designed for 3D molecular structures
        and requires a completely different input format than other models.
        It expects 3D coordinates as input, not feature vectors.
        """
        props = self.properties.copy()
        props['model_type'] = 'CNNAutoEncoder'
        props['n_cvs'] = 2

        # CNNAutoEncoder uses different architecture
        # This test is skipped because it needs specialized setup
        pass


# Summary comment for documentation
"""
Model Testing Summary
=====================

Testable with standard configuration (test_all_models.py):
- AutoEncoder: Standard feedforward architecture ✓
- VariationalAutoEncoder: Variational architecture with mean/var outputs ✓

Testable with specialized configuration (this file):
- GaussianMixtureVariationalAutoEncoder: Requires dict-based encoder/decoder ✓
- SPIB: Requires k parameter for number of states ✓

Requires extensive specialized setup:
- CNNAutoEncoder: Designed for 3D molecular coordinates, not feature vectors
  (from MoLearn framework, uses graph-based architecture)

To run all model tests:
    pytest biobb_pytorch/test/unitests/test_mdae/test_all_models.py -v
    pytest biobb_pytorch/test/unitests/test_mdae/test_specialized_models.py -v
"""
