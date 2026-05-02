from __future__ import annotations

import copy
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _resolve_path(path_value: str | None, project_root: Path) -> str | None:
    if path_value is None:
        return None

    expanded = str(path_value).replace("${project_root}", str(project_root))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = project_root / path

    return str(path.resolve())


def _resolve_config_paths(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    path_config = resolved.setdefault("paths", {})

    for key in ("train_dataset_dir", "test_dataset_dir", "output_dir"):
        if key in path_config:
            path_config[key] = _resolve_path(path_config.get(key), project_root)

    return resolved


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    default_config_path = project_root / "config" / "config.yaml"
    config_file = Path(config_path) if config_path else default_config_path
    if not config_file.is_absolute():
        config_file = (project_root / config_file).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    config = _resolve_config_paths(raw_config, project_root)
    config["runtime"] = {
        "project_root": str(project_root),
        "config_path": str(config_file),
    }
    return config


def prepare_run_directory(config: dict[str, Any]) -> Path:
    experiment_cfg = config.setdefault("experiment", {})
    path_cfg = config.setdefault("paths", {})

    output_dir = Path(path_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = experiment_cfg.get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_cfg["run_name"] = run_name

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_filename = path_cfg.get("metrics_filename", "training_metrics.csv")
    model_filename = path_cfg.get("model_filename", "TrainedModel.pt")
    resolved_config_filename = path_cfg.get("resolved_config_filename", "resolved_config.yaml")

    path_cfg["run_dir"] = str(run_dir)
    path_cfg["metrics_csv"] = str(run_dir / metrics_filename)
    path_cfg["model_path"] = str(run_dir / model_filename)
    path_cfg["resolved_config"] = str(run_dir / resolved_config_filename)

    with Path(path_cfg["resolved_config"]).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return run_dir


def resolve_device(device_config: str) -> torch.device:
    normalized = str(device_config).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but no CUDA device is available.")

    return torch.device(device_config)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
