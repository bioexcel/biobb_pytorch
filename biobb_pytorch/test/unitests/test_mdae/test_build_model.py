# type: ignore
import pytest
from biobb_common.tools import test_fixtures as fx
from biobb_pytorch.mdae.build_model import build_model, BuildModel, assert_valid_kwargs
import torch
import tempfile
from pathlib import Path


class TestBuildModel:
    def setup_class(self):
        fx.test_setup(self, 'build_model')

    def teardown_class(self):
        fx.test_teardown(self)

    def test_build_model(self):
        build_model(properties=self.properties, **self.paths)
        assert fx.not_empty(self.paths['output_model_pth_path'])

        # Load and verify model structure
        # The model is saved directly as an object, not in a dictionary
        model = torch.load(self.paths['output_model_pth_path'], weights_only=False)
        # Verify it's a PyTorch model (has state_dict method)
        assert hasattr(model, 'state_dict'), "Model file should contain a PyTorch model"
        assert hasattr(model, '_hparams'), "Model file should contain _hparams attribute"

    def test_build_model_with_custom_loss(self):
        """Test building model with custom loss function."""
        props = self.properties.copy()
        props['options'] = props.get('options', {}).copy()
        props['options']['loss_function'] = {
            'loss_type': 'MSELoss'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            build_model(properties=props, 
                       input_stats_pt_path=self.paths['input_stats_pt_path'],
                       output_model_pth_path=tmp_path)
            assert Path(tmp_path).exists()
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_build_model_no_output(self):
        """Test building model without saving."""
        instance = BuildModel(
            input_stats_pt_path=self.paths['input_stats_pt_path'],
            output_model_pth_path=None,
            properties=self.properties
        )
        assert instance.model is not None
        assert hasattr(instance.model, 'forward')

    def test_assert_valid_kwargs(self):
        """Test assert_valid_kwargs utility function."""
        class DummyClass:
            def __init__(self, a, b, c=None):
                pass
        
        # Valid kwargs should not raise
        assert_valid_kwargs(DummyClass, {'a': 1, 'b': 2}, context="test")
        assert_valid_kwargs(DummyClass, {'a': 1, 'b': 2, 'c': 3}, context="test")
        
        # Invalid kwargs should raise
        with pytest.raises(AssertionError):
            assert_valid_kwargs(DummyClass, {'a': 1, 'b': 2, 'invalid': 3}, context="test")

    def test_load_full(self):
        """Test load_full static method."""
        build_model(properties=self.properties, **self.paths)
        loaded_model = BuildModel.load_full(self.paths['output_model_pth_path'])
        assert hasattr(loaded_model, 'state_dict')
