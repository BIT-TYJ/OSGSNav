from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


def load_yaml_config(path: str) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"3DGS config must be a mapping: {path}")
    return data


def to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(v) for v in value]
    return value

