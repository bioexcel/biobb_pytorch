"""Build Optuna studies from ``properties['Optuna']``."""
from __future__ import annotations

from typing import Any, Dict, Optional

import optuna
from optuna import pruners
from optuna import samplers


def _build_sampler(spec: Optional[Dict[str, Any]]):
    if not spec:
        return None
    name = spec.get("type")
    if not name:
        return None
    cls = getattr(samplers, name, None)
    if cls is None:
        raise KeyError(f"Unknown Optuna sampler type: {name}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kwargs)


def _build_pruner(spec: Optional[Dict[str, Any]]):
    if not spec:
        return None
    name = spec.get("type")
    if not name:
        return None
    cls = getattr(pruners, name, None)
    if cls is None:
        raise KeyError(f"Unknown Optuna pruner type: {name}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kwargs)


def create_study_from_config(optuna_cfg: Dict[str, Any]) -> optuna.Study:
    direction = optuna_cfg.get("direction", "minimize")
    if direction not in ("minimize", "maximize"):
        raise ValueError("Optuna.direction must be 'minimize' or 'maximize'")

    sampler = _build_sampler(optuna_cfg.get("sampler"))
    pruner = _build_pruner(optuna_cfg.get("pruner"))
    storage = optuna_cfg.get("storage")
    study_name = optuna_cfg.get("study_name", "biobb_mdae_hparam")

    kwargs: Dict[str, Any] = {"direction": direction}
    if sampler is not None:
        kwargs["sampler"] = sampler
    if pruner is not None:
        kwargs["pruner"] = pruner

    if storage:
        kwargs["storage"] = storage
        kwargs["study_name"] = study_name
        kwargs["load_if_exists"] = bool(optuna_cfg.get("load_if_exists", True))
        return optuna.create_study(**kwargs)

    return optuna.create_study(**kwargs)
