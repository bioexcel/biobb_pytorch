# type: ignore
"""
Comprehensive test suite for all model types in biobb_pytorch.mdae.models

This test file focuses on models that can be instantiated with standard
list-based encoder/decoder layer configurations (AutoEncoder, VAE).

Models with specialized configurations (GMVAE, SPIB, CNNAutoEncoder) require
dictionaries for encoder/decoder layers and have their own specific tests.
"""
import pytest
import torch
import tempfile
from pathlib import Path
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.build_model import buildModel, BuildModel


class TestAllModels:
    """Test suite for model architectures with standard configurations."""
    
    def setup_class(self):
        """Setup test fixtures using buildModel configuration."""
        fx.test_setup(self, 'buildModel')

    def teardown_class(self):
        """Cleanup after tests."""
        fx.test_teardown(self)

    @pytest.mark.parametrize("model_type,extra_props,expected_attrs", [
        ('AutoEncoder', {}, ['encoder', 'decoder', 'norm_in']),
        ('VariationalAutoEncoder', {}, ['encoder', 'decoder', 'mean_nn', 'log_var_nn']),
    ])
    def test_build_all_model_types(self, model_type, extra_props, expected_attrs):
        """
        Test building all supported model types.
        
        Args:
            model_type: Name of the model class to instantiate
            extra_props: Additional properties specific to the model type
            expected_attrs: Expected attributes that should exist in the model
        """
        props = self.properties.copy()
        props['model_type'] = model_type
        props.update(extra_props)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Build the model
            buildModel(
                properties=props,
                input_stats_pt_path=self.paths['input_stats_pt_path'],
                output_model_pth_path=tmp_path
            )
            
            # Verify file was created
            assert Path(tmp_path).exists(), f"Model file should exist for {model_type}"
            
            # Load and verify model
            model = torch.load(tmp_path, weights_only=False)
            
            # Verify model class name
            assert model.__class__.__name__ == model_type, \
                f"Model class should be {model_type}, got {model.__class__.__name__}"
            
            # Verify basic PyTorch model properties
            assert hasattr(model, 'state_dict'), \
                f"{model_type} should have state_dict method"
            assert hasattr(model, 'forward'), \
                f"{model_type} should have forward method"
            assert hasattr(model, '_hparams'), \
                f"{model_type} should have _hparams attribute"
            
            # Verify model-specific attributes
            for attr in expected_attrs:
                assert hasattr(model, attr), \
                    f"{model_type} should have {attr} attribute"
            
            # Verify model can be put in eval mode
            model.eval()
            
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    @pytest.mark.parametrize("model_type", [
        'AutoEncoder',
        'VariationalAutoEncoder',
    ])
    def test_model_forward_pass(self, model_type):
        """
        Test that each model type can perform a forward pass.
        
        Args:
            model_type: Name of the model class to test
        """
        props = self.properties.copy()
        props['model_type'] = model_type
        
        # Build model without saving
        instance = BuildModel(
            input_stats_pt_path=self.paths['input_stats_pt_path'],
            output_model_pth_path=None,
            properties=props
        )
        
        model = instance.model
        assert model is not None, f"{model_type} should be instantiated"
        
        # Load stats to get input dimensions
        stats = torch.load(self.paths['input_stats_pt_path'], weights_only=False)
        n_features = stats['shape'][1]
        
        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, n_features)
        
        # Set model to eval mode
        model.eval()
        
        # Perform forward pass
        with torch.no_grad():
            try:
                output = model(dummy_input)
                
                # Verify output - AutoEncoder and VAE both return dict
                if isinstance(output, dict):
                    assert 'z' in output, \
                        f"{model_type} output dict should contain latent representation 'z'"
                    assert output['z'].shape[0] == batch_size, \
                        f"{model_type} latent batch size should match input"
                else:
                    # Some models might return tensor directly
                    assert output.shape[0] == batch_size, \
                        f"{model_type} output batch size should match input"
                    
            except Exception as e:
                pytest.fail(f"{model_type} forward pass failed: {str(e)}")

    @pytest.mark.parametrize("model_type,custom_layers", [
        ('AutoEncoder', {'encoder_layers': [32, 16, 8], 'decoder_layers': [8, 16, 32]}),
        ('VariationalAutoEncoder', {'encoder_layers': [32, 16], 'decoder_layers': [16, 32]}),
    ])
    def test_custom_layer_configurations(self, model_type, custom_layers):
        """
        Test building models with custom layer configurations.
        
        Args:
            model_type: Name of the model class to test
            custom_layers: Custom encoder and decoder layer configurations
        """
        props = self.properties.copy()
        props['model_type'] = model_type
        props.update(custom_layers)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            buildModel(
                properties=props,
                input_stats_pt_path=self.paths['input_stats_pt_path'],
                output_model_pth_path=tmp_path
            )
            
            assert Path(tmp_path).exists(), \
                f"Model with custom layers should be created for {model_type}"
            
            model = torch.load(tmp_path, weights_only=False)
            assert model.__class__.__name__ == model_type
            
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    @pytest.mark.parametrize("model_type", [
        'AutoEncoder',
        'VariationalAutoEncoder',
    ])
    def test_model_with_custom_loss(self, model_type):
        """
        Test building models with custom loss functions.
        
        Args:
            model_type: Name of the model class to test
        """
        props = self.properties.copy()
        props['model_type'] = model_type
        props['options'] = props.get('options', {}).copy()
        props['options']['loss_function'] = {
            'loss_type': 'MSELoss'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            buildModel(
                properties=props,
                input_stats_pt_path=self.paths['input_stats_pt_path'],
                output_model_pth_path=tmp_path
            )
            
            assert Path(tmp_path).exists(), \
                f"Model with custom loss should be created for {model_type}"
            
            model = torch.load(tmp_path, weights_only=False)
            assert hasattr(model, '_hparams'), \
                f"{model_type} should contain hparams with loss configuration"
            
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_model_state_dict_compatibility(self):
        """Test that all models produce compatible state dicts."""
        model_types = [
            'AutoEncoder',
            'VariationalAutoEncoder',
        ]
        
        for model_type in model_types:
            props = self.properties.copy()
            props['model_type'] = model_type
            
            instance = BuildModel(
                input_stats_pt_path=self.paths['input_stats_pt_path'],
                output_model_pth_path=None,
                properties=props
            )
            
            model = instance.model
            state_dict = model.state_dict()
            
            # Verify state dict contains parameters
            assert len(state_dict) > 0, \
                f"{model_type} state_dict should contain parameters"
            
            # Verify all parameters are tensors
            for key, value in state_dict.items():
                assert isinstance(value, torch.Tensor), \
                    f"All state_dict values should be tensors in {model_type}"

    @pytest.mark.parametrize("n_cvs", [1, 2, 5, 10])
    def test_different_latent_dimensions(self, n_cvs):
        """
        Test AutoEncoder with different latent space dimensions.
        
        Args:
            n_cvs: Number of collective variables (latent dimensions)
        """
        props = self.properties.copy()
        props['model_type'] = 'AutoEncoder'
        props['n_cvs'] = n_cvs
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            buildModel(
                properties=props,
                input_stats_pt_path=self.paths['input_stats_pt_path'],
                output_model_pth_path=tmp_path
            )
            
            assert Path(tmp_path).exists(), \
                f"Model should be created with n_cvs={n_cvs}"
            
            model = torch.load(tmp_path, weights_only=False)
            assert model._hparams.get('n_cvs') == n_cvs, \
                f"Model should have n_cvs={n_cvs} in hparams"
            
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_vae_encode_decode(self):
        """Test that VariationalAutoEncoder produces proper encode/decode output."""
        props = self.properties.copy()
        props['model_type'] = 'VariationalAutoEncoder'
        
        instance = BuildModel(
            input_stats_pt_path=self.paths['input_stats_pt_path'],
            output_model_pth_path=None,
            properties=props
        )
        
        model = instance.model
        stats = torch.load(self.paths['input_stats_pt_path'], weights_only=False)
        n_features = stats['shape'][1]
        n_cvs = self.properties.get('n_cvs', 2)
        
        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, n_features)
        
        model.eval()
        with torch.no_grad():
            # Test encode_decode method which is specific to VAE
            z, mean, log_var, x_hat = model.encode_decode(dummy_input)
            
            # Verify outputs
            assert z.shape == (batch_size, n_cvs), \
                f"Latent z should have shape ({batch_size}, {n_cvs})"
            assert mean.shape == (batch_size, n_cvs), \
                f"Mean should have shape ({batch_size}, {n_cvs})"
            assert log_var.shape == (batch_size, n_cvs), \
                f"Log variance should have shape ({batch_size}, {n_cvs})"
            assert x_hat.shape == (batch_size, n_features), \
                f"Reconstruction should have shape ({batch_size}, {n_features})"

