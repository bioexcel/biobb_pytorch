"""Deep copy and dot-path assignment for merging Optuna suggestions into properties."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, MutableMapping


def deep_copy_properties(props: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(props))


def set_path(root: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Set ``root["a"]["b"]["c"]`` from path ``"a.b.c"``. Creates missing dict segments."""
    parts = path.split(".")
    cur: MutableMapping[str, Any] = root
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value


def merge_suggestions(
    base: Mapping[str, Any],
    suggestions: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a deep copy of ``base`` with each ``suggestions`` dot-path applied."""
    out = deep_copy_properties(base)
    for path, value in suggestions.items():
        set_path(out, path, value)
    return out


def strip_optuna_section(props: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove ``Optuna`` key (and optionally other tuning-only keys) for ``TrainModel``."""
    out = deep_copy_properties(props)
    out.pop("Optuna", None)
    return out


def validate_search_space(search_space: Any) -> List[Dict[str, Any]]:
    if not isinstance(search_space, list):
        raise TypeError("properties['Optuna']['search_space'] must be a list")
    for i, spec in enumerate(search_space):
        if not isinstance(spec, dict):
            raise TypeError(f"search_space[{i}] must be a dict")
        if "param" not in spec or "type" not in spec:
            raise KeyError(f"search_space[{i}] requires 'param' and 'type'")
    return search_space
