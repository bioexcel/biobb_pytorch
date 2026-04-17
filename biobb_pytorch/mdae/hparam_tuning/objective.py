"""Per-trial training objective for Optuna (file-based, no in-memory API)."""
from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, Mapping

import numpy as np
import optuna
import torch

from biobb_pytorch.mdae.hparam_tuning.merge_props import merge_suggestions, strip_optuna_section
from biobb_pytorch.mdae.hparam_tuning.suggest import suggest_params
from biobb_pytorch.mdae.train_model import TrainModel


def extract_objective_value(
    metrics: Mapping[str, Any],
    objective_cfg: Mapping[str, Any],
) -> float:
    """Reduce ``metrics[metric]`` to a scalar using ``aggregation``."""
    key = objective_cfg.get("metric", "train_loss")
    aggregation = objective_cfg.get("aggregation", "last")

    if key not in metrics:
        raise KeyError(
            f"Objective metric {key!r} not in training metrics. Available: {list(metrics.keys())}"
        )
    v = metrics[key]
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    arr = np.asarray(v, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    if aggregation == "last":
        return float(arr[-1])
    if aggregation == "min":
        return float(np.min(arr))
    if aggregation == "mean":
        return float(np.mean(arr))
    if aggregation == "max":
        return float(np.max(arr))
    raise ValueError(f"Unknown aggregation: {aggregation!r}")


def make_objective(
    *,
    base_properties: Dict[str, Any],
    search_space: list,
    input_model_pth_path: str,
    input_dataset_pt_path: str,
    trials_root: str,
    optuna_cfg: Mapping[str, Any],
) -> Callable[[optuna.Trial], float]:
    objective_block = dict(optuna_cfg.get("objective") or {})

    def objective(trial: optuna.Trial) -> float:
        suggestions = suggest_params(trial, search_space)
        merged = merge_suggestions(base_properties, suggestions)
        train_props = strip_optuna_section(merged)

        trial_dir = os.path.join(trials_root, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        ckpt = os.path.join(trial_dir, "model.pth")

        tm = TrainModel(
            properties=train_props,
            input_model_pth_path=input_model_pth_path,
            input_dataset_pt_path=input_dataset_pt_path,
            output_model_pth_path=ckpt,
            output_metrics_npz_path=None,
            input_model=None,
            input_dataset=None,
        )
        try:
            tm.run_training()
            tm.save_full(tm.model)
            value = extract_objective_value(tm.metrics, objective_block)
        except Exception:
            if os.path.isdir(trial_dir):
                shutil.rmtree(trial_dir, ignore_errors=True)
            raise

        trial.set_user_attr("checkpoint", ckpt)
        trial.set_user_attr("trial_dir", trial_dir)
        return value

    return objective
