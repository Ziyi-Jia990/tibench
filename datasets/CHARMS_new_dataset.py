import csv
from typing import List, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations import Normalize  # <--- 添加
from albumentations.pytorch import ToTensorV2  # <--- 添加

def _ensure_hwc3(image, **kwargs):
        if image is None: return image
        if image.ndim == 2:   # H x W
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1: # H x W x 1
            image = np.repeat(image, 3, axis=2)
        return image

# [新增] 辅助函数：Torchvision 转换
def _to_tensor_if_numpy(x):
    if isinstance(x, np.ndarray):
        import torch
        t = torch.from_numpy(x)
        if t.ndim == 2: t = t.unsqueeze(-1)
        if t.shape[-1] == 1: t = t.repeat(1, 1, 3)
        return t.permute(2, 0, 1).contiguous().float().div(255.0)
    return x

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
                self.transform = A.Compose([
                  A.HorizontalFlip(p=0.5),
                  A.Rotate(limit=45),
                  A.ToRGB(p=1.0), # 修复 permute 错误
                  A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                  A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0)),
                  A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0), # 修复 ByteTensor
                  ToTensorV2() # 修复 ByteTensor
              ])
            elif self.target in ['pneumonia', 'los', 'rr']:
                self.transform = A.Compose([
                    # 1. [关键修复] 强制转为 HWC 3通道！防止单通道报错
                    A.Lambda(name="ensure_hwc3", image=_ensure_hwc3),
                    
                    # 2. [关键修复] 使用 RandomResizedCrop 代替 Pad+Affine
                    # 这是训练阶段最标准的增强，既改变了尺寸又增加了变异，且保证输出绝对是 img_size
                    A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
                    
                    # 其他增强保持不变
                    A.Affine(rotate=(-10, 10), translate_percent=(-0.02, 0.02), shear=(-5, 5), p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.15, p=0.6),
                    A.RandomGamma(gamma_limit=(85, 115), p=0.3),
                    A.GaussNoise(var_limit=(1.0, 10.0), p=0.2),
                    A.GaussianBlur(blur_limit=(3, 3), p=0.1),
                    A.CoarseDropout(max_holes=6, max_height=int(img_size * 0.07), max_width=int(img_size * 0.07),
                                    min_holes=1, fill_value=0, p=0.15),

                    # Normalize (针对 3 通道)
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                max_pixel_value=255.0),
                    
                    # 转 Tensor
                    ToTensorV2()
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

        # ----------------- 修改开始 -----------------
        # 判断 self.transform 是 Albumentations 还是 Torchvision
        if isinstance(self.transform, A.Compose):
            # Albumentations: 需要传入 numpy 数组，且必须使用关键字参数 image=...
            # 返回的是字典，需要取 ["image"]
            image = self.transform(image=np.array(im_pil))["image"]
        else:
            # Torchvision: 直接传入 PIL 图片，返回处理后的 Tensor
            image = self.transform(im_pil)
        # ----------------- 修改结束 -----------------

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
