import copy
import csv
from pathlib import Path

import torch
from timm.utils import adaptive_clip_grad
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils.focal_loss import FocalLoss


def train_model(model, train_loader, val_loader, DEVICE, train_config, metrics_csv_path, wandb_run=None):
    num_epochs = int(train_config.get("num_epochs", 5))
    learning_rate = float(train_config.get("learning_rate", 3e-4))
    weight_decay = float(train_config.get("weight_decay", 0.01))
    eta_min = float(train_config.get("eta_min", 1e-6))
    grad_clip_factor = float(train_config.get("grad_clip_factor", 0.01))
    amp_enabled = bool(train_config.get("amp", True)) and DEVICE.type == "cuda"

    criterion = FocalLoss(task_type="multi-class", reduction="mean")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_acc = 0
    weights = None

    metrics_path = Path(metrics_csv_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0

    with metrics_path.open(mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])

        for epoch in range(1, num_epochs + 1):
            model.train()

            train_loss = 0
            correct = 0
            total = 0

            loop = tqdm(train_loader, leave=True)
            for images, labels in loop:
                optimizer.zero_grad(set_to_none=True)

                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with autocast(device_type=DEVICE.type, enabled=amp_enabled):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                if grad_clip_factor > 0:
                    scaler.unscale_(optimizer)
                    adaptive_clip_grad(params, clip_factor=grad_clip_factor)

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loop.set_description(f"Epoch [{epoch}]")
                loop.set_postfix(loss=train_loss / total)
            if total == 0:
                raise RuntimeError("Train loader returned zero samples.")
            train_acc = correct / total
            train_loss = train_loss / total

            # validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0

            with torch.no_grad():

                for images, labels in val_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    with autocast(device_type=DEVICE.type, enabled=amp_enabled):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            if total == 0:
                raise RuntimeError("Validation loader returned zero samples.")

            val_acc = correct / total
            val_loss = val_loss / total
            current_lr = optimizer.param_groups[0]["lr"]

            loop.write(f"Epoch {epoch}/{num_epochs}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                weights = copy.deepcopy(model.state_dict())
                print("Best accuracy improved to: {:.2f}".format(best_acc * 100))

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "train/acc": train_acc,
                        "val/acc": val_acc,
                        "train/lr": current_lr,
                    },
                    step=epoch,
                )

            scheduler.step()
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, current_lr])
            f.flush()

    print(f"Best accuracy: {best_acc * 100:.2f}%")
    if weights is not None:
        model.load_state_dict(weights)

    return model, best_acc
