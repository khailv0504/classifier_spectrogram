import argparse

import torch
import os


torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from src.config import (
    load_config,
    prepare_run_directory,
    resolve_device,
    seed_everything,
)

from src.datasets.processing import Preprocessing
from src.train.training import train_model
from src.utils.visualize import Visualize

from src.module.waveCNN import WaveCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train WaveCNN with reproducible config")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config yaml. Defaults to classifier_spectrogram/config/config.yaml",
    )
    return parser.parse_args()


def init_wandb(config):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled in config but package is not installed.") from exc
