# type: ignore
import pytest
import numpy as np
import pandas as pd
from biobb_pytorch.mdae.explainability.sensitivity_analysis import make_mi_scores


class TestSensitivityAnalysis:
    def test_make_mi_scores_basic(self):
        """Test make_mi_scores with basic data."""
        # Create simple test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Create DataFrame with some correlation
        X_data = np.random.randn(n_samples, n_features)
        y = X_data[:, 0] * 2 + X_data[:, 1] + np.random.randn(n_samples) * 0.1
        
        X = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(n_features)])
        
        mi_scores = make_mi_scores(X, y, discrete_features=False)
        
        # Check output type and shape
        assert isinstance(mi_scores, pd.Series)
        assert len(mi_scores) == n_features
        assert mi_scores.name == "MI Scores"
        
        # Check that scores are sorted in descending order
        assert (mi_scores.values[:-1] >= mi_scores.values[1:]).all()
        
        # Check that all scores are non-negative
        assert (mi_scores.values >= 0).all()

    def test_make_mi_scores_discrete(self):
        """Test make_mi_scores with discrete features."""
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        
        # Create DataFrame with discrete (categorical) and continuous features
        # For discrete features, we need actual discrete values
        X_data = np.random.randn(n_samples, n_features)
        # Make first feature discrete by converting to integers
        X_data[:, 0] = np.random.randint(0, 5, n_samples)
        y = X_data[:, 0] + np.random.randn(n_samples) * 0.1
        
        X = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(n_features)])
        
        # Mark first feature as discrete (it has integer categorical values)
        discrete_features = [True, False, False]
        
        mi_scores = make_mi_scores(X, y, discrete_features=discrete_features)
        
        # Check output
        assert isinstance(mi_scores, pd.Series)
        assert len(mi_scores) == n_features

    def test_make_mi_scores_column_names(self):
        """Test that make_mi_scores preserves column names."""
        np.random.seed(42)
        n_samples = 50
        
        feature_names = ['temperature', 'pressure', 'volume']
        X_data = np.random.randn(n_samples, len(feature_names))
        y = X_data[:, 0] + np.random.randn(n_samples) * 0.1
        
        X = pd.DataFrame(X_data, columns=feature_names)
        
        mi_scores = make_mi_scores(X, y, discrete_features=False)
        
        # Check that all feature names are in the index
        for name in feature_names:
            assert name in mi_scores.index

