from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


class Visualize:
    def __init__(
        self,
        val_loader,
        model,
        DEVICE,
        class_names,
        save_dir,
    ):
        self.all_labels = []
        self.all_predictions = []
        self.class_names = class_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        with torch.no_grad():
            progress = tqdm(val_loader, leave=True)
            for images, labels in progress:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                self.all_labels.extend(labels.cpu().numpy())
                self.all_predictions.extend(predicted.cpu().numpy())

    def display_confusion_matrix(self, threshold=3):
        label_indices = list(range(len(self.class_names)))
        cm = confusion_matrix(self.all_labels, self.all_predictions, labels=label_indices)
        cm_norm = (
            confusion_matrix(
                self.all_labels,
                self.all_predictions,
                labels=label_indices,
                normalize="true",
            )
            * 100
        )

        plt.figure(figsize=(8, 6))
        ax1 = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix", pad=20, fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cbar1 = ax1.collections[0].colorbar
        cbar1.set_label("Count", rotation=0, labelpad=20, verticalalignment="center")
        plt.savefig(self.save_dir / "raw_confusion_matrix.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(8, 6))
        ax2 = sns.heatmap(
            cm_norm,
            annot=np.where(cm_norm < threshold, "", np.round(cm_norm, 2)),
            fmt="",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            vmin=0,
            vmax=100,
        )
        plt.title("Normalized Confusion Matrix", pad=20, fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cbar2 = ax2.collections[0].colorbar
        cbar2.set_label("Accuracy (%)", rotation=0, labelpad=-40, y=1.05, ha="left")
        plt.savefig(self.save_dir / "confusion_matrix.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def display_curve(self, metrics_csv_path):
        df = pd.read_csv(metrics_csv_path)
        train_loss = df["train_loss"]
        val_loss = df["val_loss"]
        train_acc = df["train_acc"]
        val_acc = df["val_acc"]
        epochs = range(1, len(train_loss) + 1)

        plt.figure()
        plt.plot(
            epochs,
            train_loss,
            label="Train Loss",
            marker="o",
            markersize=4,
            markevery=10,
            linestyle="-",
            linewidth=1.5,
        )
        plt.plot(
            epochs,
            val_loss,
            label="Validation Loss",
            marker="s",
            markersize=4,
            markevery=10,
            linestyle="-",
            linewidth=1.5,
        )

        min_train_epoch = np.argmin(train_loss) + 1
        min_val_epoch = np.argmin(val_loss) + 1

        plt.scatter(min_train_epoch, train_loss[min_train_epoch - 1], color="blue", zorder=5)
        plt.scatter(min_val_epoch, val_loss[min_val_epoch - 1], color="orange", zorder=5)

        plt.annotate(
            f"{train_loss[min_train_epoch - 1]:.4f}",
            (min_train_epoch, train_loss[min_train_epoch - 1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )
        plt.annotate(
            f"{val_loss[min_val_epoch - 1]:.4f}",
            (min_val_epoch, val_loss[min_val_epoch - 1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(self.save_dir / "loss_curve.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        plt.figure()
        plt.plot(
            epochs,
            train_acc,
            label="Train Accuracy",
            marker="o",
            markersize=4,
            markevery=10,
            linestyle="-",
            linewidth=1.5,
        )
        plt.plot(
            epochs,
            val_acc,
            label="Validation Accuracy",
            marker="s",
            markersize=4,
            markevery=10,
            linestyle="-",
            linewidth=1.5,
        )

        max_train_epoch = np.argmax(train_acc) + 1
        max_val_epoch = np.argmax(val_acc) + 1

        plt.scatter(max_train_epoch, train_acc[max_train_epoch - 1], color="blue", zorder=5)
        plt.scatter(max_val_epoch, val_acc[max_val_epoch - 1], color="orange", zorder=5)

        plt.annotate(
            f"{train_acc[max_train_epoch - 1]:.4f}",
            (max_train_epoch, train_acc[max_train_epoch - 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            f"{val_acc[max_val_epoch - 1]:.4f}",
            (max_val_epoch, val_acc[max_val_epoch - 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(self.save_dir / "accuracy_curve.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def display_report(self):
        return classification_report(
            self.all_labels,
            self.all_predictions,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
        )
