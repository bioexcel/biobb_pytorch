# type: ignore
import numpy as np
import optuna
import pytest
from optuna.study import StudyDirection

from biobb_pytorch.mdae.hparam_tuning.merge_props import (
    merge_suggestions,
    set_path,
    strip_optuna_section,
    validate_search_space,
)
from biobb_pytorch.mdae.hparam_tuning.objective import extract_objective_value
from biobb_pytorch.mdae.hparam_tuning.study_factory import create_study_from_config
from biobb_pytorch.mdae.hparam_tuning.suggest import suggest_params


class TestMergeProps:
    def test_set_path_nested(self):
        root = {"Trainer": {}}
        set_path(root, "Trainer.max_epochs", 7)
        assert root["Trainer"]["max_epochs"] == 7

    def test_merge_suggestions(self):
        base = {"Trainer": {"max_epochs": 1}, "Dataset": {"batch_size": 8}}
        merged = merge_suggestions(base, {"Trainer.max_epochs": 99, "Dataset.batch_size": 64})
        assert merged["Trainer"]["max_epochs"] == 99
        assert merged["Dataset"]["batch_size"] == 64
        assert base["Trainer"]["max_epochs"] == 1

    def test_strip_optuna(self):
        p = {"Trainer": {}, "Optuna": {"n_trials": 3}}
        s = strip_optuna_section(p)
        assert "Optuna" not in s
        assert "Trainer" in s


class TestValidateSearchSpace:
    def test_valid(self):
        ss = [{"param": "Trainer.max_epochs", "type": "int", "low": 1, "high": 5}]
        assert validate_search_space(ss) == ss

    def test_invalid(self):
        with pytest.raises(TypeError):
            validate_search_space({})


class TestSuggestParams:
    def test_suggest_float_int_categorical(self):
        study = optuna.create_study()
        trial = study.ask()
        space = [
            {"param": "Trainer.max_epochs", "type": "int", "low": 1, "high": 3},
            {"param": "Dataset.batch_size", "type": "categorical", "choices": [16, 32]},
            {"param": "Trainer.gradient_clip_val", "type": "float", "low": 0.1, "high": 1.0},
        ]
        out = suggest_params(trial, space)
        assert set(out.keys()) == {
            "Trainer.max_epochs",
            "Dataset.batch_size",
            "Trainer.gradient_clip_val",
        }
        assert out["Dataset.batch_size"] in (16, 32)
        assert 1 <= out["Trainer.max_epochs"] <= 3


class TestExtractObjectiveValue:
    def test_last_scalar(self):
        v = extract_objective_value(
            {"train_loss": np.array([1.0, 0.5])},
            {"metric": "train_loss", "aggregation": "last"},
        )
        assert v == 0.5

    def test_mean(self):
        v = extract_objective_value(
            {"train_loss": np.array([1.0, 3.0])},
            {"metric": "train_loss", "aggregation": "mean"},
        )
        assert v == 2.0


class TestStudyFactory:
    def test_create_study_defaults(self):
        study = create_study_from_config({})
        assert study.direction == StudyDirection.MINIMIZE

    def test_tpe_sampler(self):
        study = create_study_from_config(
            {
                "direction": "maximize",
                "sampler": {"type": "TPESampler", "seed": 0},
            }
        )
        assert study.direction == StudyDirection.MAXIMIZE
