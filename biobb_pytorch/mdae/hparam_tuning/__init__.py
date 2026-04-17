"""Optuna-based hyperparameter tuning for ``TrainModel`` (path-based I/O)."""

from biobb_pytorch.mdae.hparam_tuning.tuner import HparamTuning, hparam_tuning, main
from biobb_pytorch.mdae.hparam_tuning.merge_props import (
    deep_copy_properties,
    merge_suggestions,
    set_path,
    strip_optuna_section,
)
from biobb_pytorch.mdae.hparam_tuning.study_factory import create_study_from_config
from biobb_pytorch.mdae.hparam_tuning.suggest import suggest_params
from biobb_pytorch.mdae.hparam_tuning.objective import extract_objective_value, make_objective

__all__ = [
    "HparamTuning",
    "hparam_tuning",
    "main",
    "create_study_from_config",
    "deep_copy_properties",
    "extract_objective_value",
    "make_objective",
    "merge_suggestions",
    "set_path",
    "strip_optuna_section",
    "suggest_params",
]
