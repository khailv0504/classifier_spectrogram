import argparse

import torch
import os

from torch import nn

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

    run_name = wandb_cfg.get("run_name") or config.get("experiment", {}).get("run_name")
    return wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=run_name,
        config=config,
        mode=wandb_cfg.get("mode", "offline"),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        dir=config["paths"]["run_dir"],
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    run_dir = prepare_run_directory(config)

    experiment_cfg = config.get("experiment", {})
    seed = int(experiment_cfg.get("seed", 42))
    deterministic = bool(experiment_cfg.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    device = resolve_device(experiment_cfg.get("device", "auto"))
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")

    root_dir = config["paths"]["train_dataset_dir"]
    pipeline_preprocessing = Preprocessing(
        root_dir=root_dir,
        data_config=config.get("data", {}),
        seed=seed,
    )
    train_loader, val_loader, class_names = pipeline_preprocessing.process()
    print("Classes:", class_names)

    model_cfg = config.get("model", {})
    num_classes = int(model_cfg.get("num_classes", len(class_names)))

    model = torch.jit.load(r"D:\deep_learning\classifier_spectrogram\src\output\20260502_080655\TrainedModel.pt")
    for p in model.parameters():
        p.requires_grad = False
    
    in_features = model.head[3].in_features
    model.head[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    print("Params: ", sum(p.numel() for p in model.parameters()))

    wandb_run = init_wandb(config)
    try:
        model, best_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            DEVICE=device,
            train_config=config.get("train", {}),
            metrics_csv_path=config["paths"]["metrics_csv"],
            wandb_run=wandb_run,
        )
        try:
            visualize_cfg = config.get("visualize", {})
            if visualize_cfg.get("enabled", True):
                visualizing = Visualize(
                    val_loader=val_loader,
                    model=model,
                    DEVICE=device,
                    class_names=class_names,
                    save_dir=config["paths"]["run_dir"],
                )
                print(visualizing.display_report())
                visualizing.display_curve(config["paths"]["metrics_csv"])
                visualizing.display_confusion_matrix(
                    threshold=float(visualize_cfg.get("confusion_threshold", 3))
                )
        except KeyboardInterrupt:
            print("Interrupted by user")

        model.eval()
        example_input = torch.randn(1, 3, 224, 224).to(device)
        traced_model = torch.jit.trace(model, example_input)
        model_path = config["paths"]["model_path"]
        traced_model.save(model_path)
        print("Model saved:", model_path)

        if wandb_run is not None:
            wandb_run.summary["best_val_acc"] = best_acc
            wandb_run.summary["model_path"] = model_path
            wandb_run.summary["metrics_csv"] = config["paths"]["metrics_csv"]
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()