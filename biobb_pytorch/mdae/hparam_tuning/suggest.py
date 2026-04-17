"""Map declarative search_space entries to Optuna trial suggestions."""
from __future__ import annotations

from typing import Any, Dict, List

import optuna

from biobb_pytorch.mdae.hparam_tuning.merge_props import validate_search_space


def suggest_params(trial: optuna.Trial, search_space: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For each spec, call ``trial.suggest_*`` and return ``{dot_path: value}``.

    Supported ``type`` values: ``float``, ``int``, ``categorical``.
    For ``float``: ``low``, ``high``, optional ``log`` (bool).
    For ``int``: ``low``, ``high``, optional ``log`` (bool).
    For ``categorical``: ``choices`` (list).
    """
    validate_search_space(search_space)
    out: Dict[str, Any] = {}
    for spec in search_space:
        param_path = spec["param"]
        stype = spec["type"]
        # Optuna parameter names must be unique strings without special issues
        pname = f"p_{len(out)}_{param_path.replace('.', '__')}"

        if stype == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))
            val = trial.suggest_float(pname, low, high, log=log)
        elif stype == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            log = bool(spec.get("log", False))
            val = trial.suggest_int(pname, low, high, log=log)
        elif stype == "categorical":
            choices = list(spec["choices"])
            if not choices:
                raise ValueError(f"categorical choices empty for {param_path}")
            val = trial.suggest_categorical(pname, choices)
        else:
            raise ValueError(f"Unsupported search_space type: {stype!r} for {param_path}")

        out[param_path] = val
    return out
