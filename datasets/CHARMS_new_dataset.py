import csv
from typing import List, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ConCatImageDataset(Dataset):
    """
    Generic (1 row -> 1 image) ConCat dataset.

    Required inputs:
      - tabular_csv_path: headerless numeric csv, shape (N, D)
      - image_paths_pt:  torch-saved list of absolute .npy paths, length N
      - label_pt:        torch-saved labels tensor/list, length N
      - field_lengths_path: torch-saved field lengths (list/tensor), length D
          * length == 1  -> continuous feature
          * length  > 1  -> categorical feature (integer-coded in CSV)

    Returns (same structure as old PetFinderConCatImageDataset):
        (table_features_con, table_features_cat, image), label
      - table_features_con: FloatTensor, shape (#con,)
      - table_features_cat: LongTensor,  shape (#cat,)
      - image: FloatTensor, shape (3, H, W)
      - label: LongTensor for classification, FloatTensor for regression
    """

    def __init__(
        self,
        tabular_csv_path: str,
        image_paths_pt: str,
        label_pt: str,
        field_lengths_path: str,
        target: Optional[str] = None,   # e.g. dataset name, or can be 'regression'/'classification' if you prefer
        train: bool = True,
        img_size: int = 224,
        task: str = "classification",   # "classification" or "regression"
    ):
        self.tabular_csv_path = tabular_csv_path
        self.image_paths_pt = image_paths_pt
        self.label_pt = label_pt
        self.field_lengths_path = field_lengths_path
        self.target = target
        self.train = train
        self.task = task

        # ---- image transforms ----
        if train:
            if self.target == 'breast_cancer': # 或者是你的数据集名字
                # 针对医学灰度图的增强
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), # 只有轻微的裁剪，医学图像通常主体在中间
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(), # 医学图像通常垂直翻转也是合理的
                    # 注意：这里删除了 ColorJitter
                    transforms.ToTensor(),
                    # 方案B提到的归一化修改，见下文
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # ---- load tabular ----
        self.X = self._read_numeric_csv(self.tabular_csv_path)  # (N, D) float32

        # ---- load labels ----
        y = torch.load(self.label_pt, map_location="cpu")
        if isinstance(y, torch.Tensor) and y.ndim > 1:
            y = y.view(-1)
        self.labels = y

        # ---- load field lengths ----
        fl = torch.load(self.field_lengths_path, map_location="cpu")
        if isinstance(fl, torch.Tensor):
            fl = fl.detach().cpu().numpy()
        self.field_lengths = np.asarray(fl, dtype=np.int64).reshape(-1)  # (D,)

        if self.X.shape[1] != len(self.field_lengths):
            raise ValueError(
                f"Feature dim mismatch: X has {self.X.shape[1]} cols, "
                f"but field_lengths has {len(self.field_lengths)}."
            )

        # ---- con/cat indices from field_lengths ----
        self.con_indices = np.where(self.field_lengths == 1)[0]
        self.cat_indices = np.where(self.field_lengths > 1)[0]

        # ---- load image paths (.npy absolute paths) ----
        paths_obj = torch.load(self.image_paths_pt)
        self.image_paths: List[str] = [self._extract_path_from_entry(e) for e in list(paths_obj)]

        # ---- sanity checks ----
        n = self.X.shape[0]
        if len(self.image_paths) != n:
            raise ValueError(f"len(image_paths)={len(self.image_paths)} != num_rows={n}")
        if len(self.labels) != n:
            raise ValueError(f"len(labels)={len(self.labels)} != num_rows={n}")

        print(
            f"[ConCatImageDataset] N={n}, D={self.X.shape[1]}, "
            f"#con={len(self.con_indices)}, #cat={len(self.cat_indices)}, "
            f"train={self.train}, task={self.task}, target={self.target}"
        )

    # ---------------- helpers ----------------
    def _read_numeric_csv(self, path: str) -> np.ndarray:
        # fast path
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception:
            # safe fallback
            data = []
            with open(path, "r") as f:
                reader = csv.reader(f)
                for r in reader:
                    if r:
                        data.append([float(x) for x in r])
            return np.asarray(data, dtype=np.float32)

    def _extract_path_from_entry(self, entry: Any) -> str:
        # common: entry is str
        if isinstance(entry, str):
            return entry
        # common: (path, ...)
        if isinstance(entry, (tuple, list)) and len(entry) > 0 and isinstance(entry[0], str):
            return entry[0]
        # common: {"npy_path": "..."} or {"path": "..."}
        if isinstance(entry, dict):
            for k in ["npy_path", "path", "image_path", "img_path", "file", "filepath"]:
                v = entry.get(k, None)
                if isinstance(v, str):
                    return v
        return str(entry)

    def _load_npy_as_pil(self, npy_path: str) -> Image.Image:
        arr = np.load(npy_path, allow_pickle=True)

        # HW -> HWC
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        # HWC1 -> HWC3
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            # heuristic: 0..1
            if np.nanmax(arr) <= 1.5:
                arr *= 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    # ---------------- Dataset API ----------------
    def __getitem__(self, index: int):
        x = self.X[index]  # (D,)

        x_con = x[self.con_indices] if len(self.con_indices) > 0 else np.zeros((0,), dtype=np.float32)
        x_cat = x[self.cat_indices] if len(self.cat_indices) > 0 else np.zeros((0,), dtype=np.float32)

        table_features_con = torch.tensor(x_con, dtype=torch.float)
        table_features_cat = torch.tensor(x_cat.astype(np.int64), dtype=torch.long)

        npy_path = self.image_paths[index]
        im_pil = self._load_npy_as_pil(npy_path)
        image = self.transform(im_pil)

        y = self.labels[index]
        if self.task == "regression":
            label = torch.tensor(float(y), dtype=torch.float)
        else:
            label = torch.tensor(int(y), dtype=torch.long)

        return (table_features_con, table_features_cat, image), label

    def __len__(self):
        return self.X.shape[0]

    # ---- convenience getters (useful for hparams) ----
    def get_num_continuous(self) -> int:
        return int(len(self.con_indices))

    def get_num_categorical(self) -> int:
        return int(len(self.cat_indices))

    def get_num_fields(self) -> int:
        # total fields D
        return int(len(self.field_lengths))

    def get_cat_cardinalities(self) -> List[int]:
        # cardinalities for each categorical field (aligned to table_features_cat order)
        return [int(self.field_lengths[i]) for i in self.cat_indices]
