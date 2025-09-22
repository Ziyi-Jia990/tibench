import random
from typing import List, Tuple
import os
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

class ContrastiveImageDataset(Dataset):
  def __init__(self, data: str, labels: str, transform: transforms.Compose, delete_segmentation: bool, augmentation_rate: float, img_size: int, live_loading: bool) -> None:
      self.live_loading = live_loading
      self.transform = transform
      self.augmentation_rate = augmentation_rate
      
      # ❗❗❗ 这是修正的核心部分 ❗❗❗
      if self.live_loading:
          # 实时加载模式：data是文件夹路径，self.data存储图片路径列表
          print("INFO: Loading data in live_loading mode. `data` is treated as a directory.")
          # 根据你的图片格式修改 '*.jpg'
          self.data = sorted(glob.glob(os.path.join(data, '*.jpg')))
          if not self.data:
              raise FileNotFoundError(f"No images found at path: {data}")
      else:
          # 预加载模式：data是.pt文件路径，self.data存储Tensor
          print("INFO: Loading data in pre-loaded mode. `data` is treated as a .pt file.")
          self.data = torch.load(data)

      self.labels = torch.load(labels)

      if delete_segmentation and not self.live_loading:
          # 这个操作只在预加载模式下有意义，因为所有数据都在内存里
          for im in self.data:
              im[0,:,:] = 0
      # 注意：delete_segmentation 在 live_loading 模式下需要另外处理，
      # 可以在 generate_imaging_views 中应用

      self.default_transform = transforms.Compose([
          transforms.Resize(size=(img_size,img_size)),
          transforms.Lambda(lambda x : x.float())
      ])

  def __getitem__(self, indx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Returns two augmented views of one image and its label
    """
    view_1, view_2 = self.generate_imaging_views(indx)

    return view_1, view_2, self.labels[indx]

  def __len__(self) -> int:
    return len(self.data)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. 
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data[index]
    if self.live_loading:
      im = read_image(im)
      im = im / 255
    view_1 = self.transform(im)
    if random.random() < self.augmentation_rate:
      view_2 = self.transform(im)
    else:
      view_2 = self.default_transform(im)
    
    return view_1, view_2