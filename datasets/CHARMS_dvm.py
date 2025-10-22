import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from typing import Tuple, Any
import pandas as pd

class DVMCarMultimodalDataset(Dataset):
    """
    用于 DVM-Car 数据集的多模态 Dataset 类。
    - 根据 .pt 文件中存储的路径列表实时加载图像。
    - 从 CSV 文件加载【已预处理】的表格特征。
    - 在 __getitem__ 中拆分连续和类别表格特征。
    """
    def __init__(
        self,
        image_paths_pt_path: str, 
        tabular_csv_path: str,    
        label_pt_path: str,      
        img_size: int = 128,     
        train: bool = True       
    ) -> None:
        super().__init__()
        
        print(f"[Dataset] 正在加载图像路径列表: {image_paths_pt_path}")
        self.image_paths = torch.load(image_paths_pt_path) 
        
        print(f"[Dataset] 正在从 CSV 加载【已预处理】表格数据: {tabular_csv_path}")
        try:
            df_tabular = pd.read_csv(tabular_csv_path, header=None) 
            self.tabular_data = torch.tensor(df_tabular.values, dtype=torch.float32)
            print(f"[Dataset] 已通过 pandas 加载并转换 CSV。张量形状: {self.tabular_data.shape}")
        except Exception as e:
            print(f"\n[错误] 加载或处理 CSV 文件失败: {tabular_csv_path} - {e}")
            raise 

        print(f"[Dataset] 正在加载标签数据: {label_pt_path}")
        self.labels = torch.load(label_pt_path)

        # --- 健全性检查 ---
        num_paths = len(self.image_paths)
        num_tabular = len(self.tabular_data)
        num_labels = len(self.labels)
        assert num_paths == num_tabular == num_labels, \
            f"数据样本数量不匹配！图像路径: {num_paths}, 表格: {num_tabular}, 标签: {num_labels}"
        print(f"[Dataset] 数据加载完成，共 {num_labels} 个样本。")

        # --- 定义连续和类别特征索引 ---
        num_total_features = self.tabular_data.shape[1]
        if num_total_features != 17:
             print(f"[警告] 表格特征文件的列数 ({num_total_features}) 与预期的 17 不符！假定前 {num_total_features-4} 列为连续，后 4 列为类别。")
        
        num_continuous = num_total_features - 4
        self.con_cols_indices = list(range(num_continuous))
        self.cat_cols_indices = list(range(num_continuous, num_total_features))
        
        # 定义 con_cols 属性 (下游代码需要)
        self.con_cols = self.con_cols_indices 
        print(f"[Dataset] 已定义连续特征索引 (con_cols): {self.con_cols}")
        print(f"[Dataset] 已定义类别特征索引 (cat_cols_indices): {self.cat_cols_indices}")

        # 计算 cat_cardinalities (下游代码需要)
        print("[Dataset] 正在计算类别特征的基数 (cardinalities)...")
        self.cat_cardinalities = []
        categorical_data = self.tabular_data[:, self.cat_cols_indices]
        for i in range(categorical_data.shape[1]):
            max_val = torch.max(categorical_data[:, i].long())
            cardinality = max_val.item() + 1
            self.cat_cardinalities.append(cardinality)
            original_col_index = self.cat_cols_indices[i]
            print(f"  - 类别特征列 (原始索引 {original_col_index}): 最大值 = {max_val.item()}, 基数 = {cardinality}")
        print(f"[Dataset] 计算完成的类别基数 (cat_cardinalities): {self.cat_cardinalities}")

        self.train = train
        self.img_size = img_size

        # --- 定义图像变换 (保持不变) ---
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_train = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32), 
            transforms.Resize((img_size, img_size)), 
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), # 移除 antialias 
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
        self.transform_val = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32), 
            transforms.Resize((img_size, img_size)), # 移除 antialias
            normalize,
        ])
        print("[Dataset] 图像变换已定义。")


    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        根据索引获取一个数据样本，并拆分表格特征。
        """
        # --- 获取标签 ---
        label_int = self.labels[index] 
        label = torch.tensor(label_int, dtype=torch.long)          

        # --- 加载和处理图像 ---
        image_path = self.image_paths[index]
        try:
            image = read_image(image_path)
            if image.shape[0] == 1: image = image.repeat(3, 1, 1)
            
            if self.train: transformed_image = self.transform_train(image)
            else: transformed_image = self.transform_val(image)
                
        except Exception as e:
            print(f"\n[错误] 加载或处理图像失败: {image_path} - {e}")
            transformed_image = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32) 

        # =====================================================================
        # ▼▼▼ 核心修复：在这里拆分表格特征 ▼▼▼
        # =====================================================================
        # 获取完整的表格特征行
        tabular_full_features = self.tabular_data[index]
        
        # 根据索引拆分连续和类别特征
        table_features_con = tabular_full_features[self.con_cols_indices].float()
        # 类别特征通常需要 long 类型以便输入 Embedding 层
        table_features_cat = tabular_full_features[self.cat_cols_indices].long() 
        # =====================================================================

        # --- 返回数据 (现在包含三个特征元素) ---
        modalities = (table_features_con, table_features_cat, transformed_image)
        return modalities, label

    def __len__(self) -> int:
        return len(self.labels)