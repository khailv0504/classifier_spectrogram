import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from classifier_spectrogram.src.classifier_spectrogram.config import load_config, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate exported TorchScript model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config yaml. Defaults to classifier_spectrogram/config/config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to TorchScript model (.pt)",
    )
    return parser.parse_args()


def build_test_loader(config):
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})

    test_dir = paths_cfg.get("test_dataset_dir")
    if not test_dir:
        raise ValueError("paths.test_dataset_dir is missing in config.")

    mean = data_cfg.get("normalize_mean", [0.5, 0.5, 0.5])
    std = data_cfg.get("normalize_std", [0.5, 0.5, 0.5])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(eval_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    return test_loader, test_dataset.classes


def evaluate_model(model_path, test_loader, device):
    model = torch.jit.load(str(model_path))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix(acc=correct / total)

    accuracy = correct / total
    return accuracy, correct, total


def main():
    args = parse_args()
    config = load_config(args.config)
    experiment_cfg = config.get("experiment", {})

    device = resolve_device(experiment_cfg.get("device", "auto"))
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    test_loader, class_names = build_test_loader(config)
    print("\nTest dataset loaded")
    print("Classes:", class_names)
    print("Total test samples:", len(test_loader.dataset))

    print("\n==============================")
    print("EVALUATING MODEL")
    print("==============================")
    print("Model:", model_path.name)

    accuracy, correct, total = evaluate_model(model_path, test_loader, device)
    print("\n==============================")
    print("RESULT")
    print("==============================")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Correct  : {correct}/{total}")


if __name__ == "__main__":
    main()
