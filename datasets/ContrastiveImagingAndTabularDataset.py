from typing import List, Tuple
import random
import csv
import copy

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image

import glob
import os
from collections import defaultdict
from PIL import Image

class ContrastiveImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first tabular view is never augmented. The second view is corrupted by replacing {corruption_rate} features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_tabular: str, corruption_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
      labels_path: str, img_size: int, live_loading: bool) -> None:
            
    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])

    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.generate_marginal_distributions(data_path_tabular)
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.one_hot_tabular = one_hot_tabular
    
    # Classifier
    self.labels = torch.load(labels_path)
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def generate_marginal_distributions(self, data_path: str) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data_df = pd.read_csv(data_path)
    self.marginal_distributions = data_df.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.data[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. Also returns original image resized to required dimensions.
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data_imaging[index]
    if self.live_loading:
      im = read_image(im)
      im = im / 255
    ims = [self.transform(im)]
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(im))
    else:
      ims.append(self.default_transform(im))

    orig_im = self.default_transform(im)
    
    return ims, orig_im

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    imaging_views, unaugmented_image = self.generate_imaging_views(index)
    tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
    label = torch.tensor(self.labels[index], dtype=torch.long)
    return imaging_views, tabular_views, label, unaugmented_image

  def __len__(self) -> int:
    return len(self.data_tabular)



class ContrastiveImagingAndTabularDataset_PetFinder(ContrastiveImagingAndTabularDataset):
    
  def __init__(
        self,
        data_path_imaging: str, 
        delete_segmentation: bool, 
        augmentation: transforms.Compose, 
        augmentation_rate: float,
        data_path_tabular: str, 
        corruption_rate: float, 
        field_lengths_tabular: str,
        one_hot_tabular: bool,
        labels_path: str,
        img_size: int, 
        live_loading: bool
    ):
      """
      修正了執行順序的最終版本初始化函數。
      """
      print("--- 正在執行 PetFinder 子類別的最終版初始化... ---")
      
      # 1. 初始化所有從 pretrain.py 傳入的參數
      self.data_path_imaging = data_path_imaging
      self.transform = augmentation
      self.augmentation_rate = augmentation_rate
      self.c = corruption_rate
      self.one_hot_tabular = one_hot_tabular
      self.img_size = img_size
      self.live_loading = live_loading
      
      # 2. 定義【原始的】欄位名稱【集合】，用於數據清洗
      _cat_cols_set = {'Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                        'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                        'Sterilized', 'Health'}
      _con_cols_set = {'Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'score' , 
                        'magnitude','desc_length', 'average_word_length', 'desc_words'}

      # 3. 讀取並獲得【初步】清洗後的 DataFrame (此時仍包含 PetID 和 AdoptionSpeed)
      data_df = self.read_and_parse_csv(data_path_tabular, _cat_cols_set, _con_cols_set)
      
      # 4. 【關鍵修正第一步】: 先分離標籤和ID
      self.labels = data_df['AdoptionSpeed']
      self.id = data_df['PetID']
      
      # 5. 【關鍵修正第二步】: 從 DataFrame 中【徹底移除】非特徵欄位
      self.data_tabular = data_df.drop(columns=['AdoptionSpeed', 'PetID'])
      
      # 6. 【關鍵修正第三步】: 現在，在【只剩下特徵】的 DataFrame 上，動態獲取最終的、正確的欄位順序
      self.final_ordered_columns = self.data_tabular.columns.tolist()
      self.cat_cols = [col for col in self.final_ordered_columns if col in _cat_cols_set]
      self.con_cols = [col for col in self.final_ordered_columns if col in _con_cols_set]
      print(f"動態獲取的【純特徵】欄位順序為: {self.final_ordered_columns}")
      
      # 7. 根據【正確的】純特徵欄位順序，來構建規則
      col_to_num_classes = {
          'Type': 3, 'Breed1': 308, 'Breed2': 308, 'Gender': 4, 'Color1': 8, 'Color2': 8,
          'Color3': 8, 'MaturitySize': 5, 'FurLength': 4, 'Vaccinated': 4, 'Dewormed': 4,
          'Sterilized': 4, 'Health': 4,
      }
      for col in self.con_cols:
          col_to_num_classes[col] = 1

      field_lengths_list = [col_to_num_classes[col] for col in self.final_ordered_columns]
      self.field_lengths_tabular = torch.tensor(field_lengths_list, dtype=torch.long)
      
      # 8. 初始化圖像數據
      self.data_imaging = self.read_image_files(self.data_path_imaging)
      self.default_transform = transforms.Compose([
          transforms.Resize((self.img_size, self.img_size)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

      # 9. 初始化邊緣分佈
      self.generate_marginal_distributions()

  # 【【【 ADD THIS METHOD TO YOUR CLASS 】】】
  def generate_marginal_distributions(self) -> None:
      """
      [OVERRIDE] 
      This method overrides the parent's faulty version.
      It uses the already loaded and cleaned self.data_tabular DataFrame 
      to ensure the column order is correct.
      """
      print("--- Using CORRECTED method to generate marginal distributions from cleaned data... ---")
      # Do NOT re-read the CSV. Use the processed DataFrame.
      # We also need to re-add the PetID and AdoptionSpeed temporarily 
      # so the shape matches what the parent might expect.
      # A safer way is to build it just from the data we have.
      self.marginal_distributions = self.data_tabular.transpose().values.tolist()

  # 【修改點 6】: 覆寫 read_and_parse_csv
  # 這是解決所有數據類型和範圍錯誤的核心。
  def read_and_parse_csv(self, path_tabular: str, cat_cols_set: set, con_cols_set: set) -> pd.DataFrame:
    data = pd.read_csv(path_tabular)
      
    # 丢弃原始不需要的列
    columns_to_drop = ['RescuerID', 'Description', 'State']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # 现在，我们根据传入的列名集合来筛选出我们真正需要的列
    all_needed_cols = list(cat_cols_set | con_cols_set) + ['AdoptionSpeed', 'PetID']
    # 筛选，并保证顺序
    data = data[[col for col in data.columns if col in all_needed_cols]]

    data = data.drop(index=data[data['PhotoAmt'] == 0.0].index)
    
    for col in cat_cols_set:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

    for col in con_cols_set:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            col_data = data[col].values
            if len(col_data) > 0:
                mean, std = col_data.mean(), col_data.std()
                if std > 0:
                    data[col] = (col_data - mean) / std
    
    return data.reset_index(drop=True)

  # 【修改點 7】: 覆寫 generate_marginal_distributions
  # 修改為使用已經加載到記憶體中的 self.data_tabular，避免重複讀取文件，更高效、更準確。
  def generate_marginal_distributions(self) -> None:
      self.marginal_distributions = self.data_tabular.transpose().values.tolist()

  # 子類別特有的輔助函數
  def read_image_files(self, img_path: str) -> dict:
      image_files = list(glob.glob(os.path.join(img_path, "*.jpg")))
      image_id_dict = defaultdict(list)
      for image_file in image_files:
          file_name = os.path.basename(image_file)
          pet_id = file_name.split('-')[0].replace('.jpg', '')
          image_id_dict[pet_id].append(image_file)
      return image_id_dict

  # 【修改點 8】: 覆寫 __getitem__
  # 使其能正確處理 pandas DataFrame，並與父類的 corrupt 和 one_hot_encode 方法兼容。
  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
        # 不再需要复杂的侦错，因为数据和规则现在保证一致
        num_classes = int(self.field_lengths_tabular[i])
        if num_classes == 1:
            # 连续值
            out.append(subject[i].unsqueeze(0))
        else:
            # 类别值
            out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=num_classes))
    return torch.cat(out)
        
  def __getitem__(self, index: int) -> tuple:
    # 这个版本是正确的
    imaging_views, unaugmented_image = self.generate_imaging_views(index)
    
    row_numpy = self.data_tabular.iloc[index].values
    
    view1 = torch.tensor(row_numpy, dtype=torch.float)
    view2 = torch.tensor(self.corrupt(row_numpy.tolist()), dtype=torch.float)

    tabular_views = [view1, view2]
    
    if self.one_hot_tabular:
        tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]

    label = torch.tensor(self.labels.iloc[index], dtype=torch.long)
    
    return imaging_views, tabular_views, label, unaugmented_image

  # 子類別特有的方法
  def generate_imaging_views(self, index: int) -> tuple:
      pet_id = self.id.iloc[index]
      pet_images = self.data_imaging.get(pet_id)
      if not pet_images:
          print(f"警告：PetID {pet_id} 找不到對應圖片，將使用空白圖片代替。")
          dummy_tensor = torch.zeros((3, self.img_size, self.img_size))
          return [dummy_tensor, dummy_tensor], dummy_tensor

      im_path = random.choice(pet_images)
      im = Image.open(im_path).convert('RGB')
      
      ims = [self.transform(im)]
      if random.random() < self.augmentation_rate:
          ims.append(self.transform(im))
      else:
          ims.append(self.default_transform(im))
      
      orig_im = self.default_transform(im)
      return ims, orig_im

  def __len__(self) -> int:
    return len(self.data_tabular)