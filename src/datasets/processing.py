import glob
import os
import random
import re

import numpy as np
import torchaudio
import torchvision
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2


from classifier_spectrogram.src.datasets.spectrogram import SpectrogramDataset


class Preprocessing:
    def __init__(self, root_dir, data_config=None, seed=42):
        self.root_dir = root_dir
        self.data_config = data_config or {}
        self.seed = seed

        self.n_splits = int(self.data_config.get("n_splits", 5))
        self.fold_index = int(self.data_config.get("fold_index", 0))
        self.batch_size_train = int(self.data_config.get("batch_size_train", 64))
        self.batch_size_val = int(self.data_config.get("batch_size_val", 32))
        self.num_workers = int(self.data_config.get("num_workers", 4))
        self.pin_memory = bool(self.data_config.get("pin_memory", True))
        self.drop_last_train = bool(self.data_config.get("drop_last_train", True))
        self.normalize_mean = self.data_config.get("normalize_mean", [0.5, 0.5, 0.5])
        self.normalize_std = self.data_config.get("normalize_std", [0.5, 0.5, 0.5])
        self.time_mask_param = int(self.data_config.get("time_mask_param", 15))
        self.freq_mask_param = int(self.data_config.get("freq_mask_param", 20))

    def parse_info(self, path):
        filename = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path))  # folder name

        snr = int(re.search(r"snr(\d+)", filename).group(1))

        group_id = f"{label}_snr{snr}"

        return label, snr, group_id

    @staticmethod
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def process(self):
        paths = sorted(glob.glob(os.path.join(self.root_dir, "**/*.png"), recursive=True))
        if not paths:
            raise FileNotFoundError(f"No PNG files found under dataset path: {self.root_dir}")

        labels = sorted(list(set(os.path.basename(os.path.dirname(p)) for p in paths)))
        label2idx = {l: i for i, l in enumerate(labels)}

        groups = []
        y = []

        for p in paths:
            label, snr, group_id = self.parse_info(p)
            groups.append(group_id)
            y.append(label2idx[label])

        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )
        fold_splits = list(sgkf.split(paths, y, groups))
        if self.fold_index >= len(fold_splits):
            raise ValueError(
                f"fold_index={self.fold_index} is out of range for n_splits={self.n_splits}"
            )
        train_idx, val_idx = fold_splits[self.fold_index]

        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]

        # PARSE TOÀN BỘ METADATA TRƯỚC (CHẠY 1 LẦN)
        def get_labels(path_list):
            labels = []
            for p in path_list:
                label_str, _, _= self.parse_info(p)
                labels.append(label2idx[label_str])
            return np.array(labels, dtype=np.int64)

        train_labels = get_labels(train_paths)
        val_labels = get_labels(val_paths)


        train_tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_param),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param),
            v2.Normalize(self.normalize_mean, self.normalize_std),
        ])
        val_tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            v2.Normalize(self.normalize_mean, self.normalize_std),
        ])

        # BƠM DATA TĨNH VÀO DATASET
        train_dataset = SpectrogramDataset(
            train_paths, train_labels,
            transform=train_tf
        )
        val_dataset = SpectrogramDataset(
            val_paths, val_labels,
            transform=val_tf
        )

        train_generator = torch.Generator()
        train_generator.manual_seed(self.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_train,
            worker_init_fn=self._seed_worker,
            generator=train_generator,
            persistent_workers=self.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            worker_init_fn=self._seed_worker,
            persistent_workers=self.num_workers > 0,
        )

        return train_loader, val_loader, labels
