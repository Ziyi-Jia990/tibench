from typing import List, Tuple
from os.path import join
import os
import sys
from torch import nn
from datetime import datetime

import torch
import numpy as np
import albumentations as A
from albumentations import Normalize  # <--- 添加
from albumentations.pytorch import ToTensorV2  # <--- 添加
from torchvision import transforms # <--- 确保这个也在



def create_logdir(hparams, wandb_logger):
    """
    Creates the log directory. If hparams.checkpoint_dir is specified, it uses that.
    Otherwise, it creates a dynamic directory based on datatype and run name.
    """
    # Check if a specific checkpoint directory is provided in the config
    if hasattr(hparams, 'checkpoint_dir') and hparams.checkpoint_dir:
        logdir = hparams.checkpoint_dir
        print(f"Using specified checkpoint directory: {logdir}")
        os.makedirs(logdir, exist_ok=True)
        return logdir
    
    # --- Fallback to the original logic if checkpoint_dir is not set ---
    print("checkpoint_dir not specified, creating dynamic log directory.")
    basepath = os.path.join("logs", str(hparams.datatype)) 
    os.makedirs(basepath, exist_ok=True)

    run_name = None
    if wandb_logger is not None and getattr(wandb_logger, "experiment", None) is not None:
        run_name = getattr(wandb_logger.experiment, "name", None) or \
                   getattr(wandb_logger.experiment, "id", None)

    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    logdir = os.path.join(basepath, str(run_name)) # Use os.path.join for consistency
    os.makedirs(logdir, exist_ok=True)
    return logdir

def convert_to_float(x):
  return x.float()


def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x  = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x


def grab_image_augmentations(img_size: int, target: str, augmentation_speedup: bool = False, crop_scale_lower: float = 0.08) -> transforms.Compose:
    """
    Defines augmentations to be used with images during contrastive training and creates Compose.
    """
    
    # [新增] 标准化 target 字符串
    target = target.lower()

    if target == 'dvm':
        if augmentation_speedup:
            transform = A.Compose([
                A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(blur_limit=(29,29), sigma_limit=(0.1,2.0), p=0.5),
                A.RandomResizedCrop(size=(img_size, img_size), scale=(crop_scale_lower, 1.0), ratio=(0.75, 1.3333333333333333)),
                # A.RandomResizedCrop(height=img_size, width=img_size, scale=(crop_scale_lower, 1.0), ratio=(0.75, 1.3333333333333333)),
                A.HorizontalFlip(p=0.5),
                A.Lambda(name='convert2tensor', image=convert_to_ts) # [0, 255] -> [0, 1]
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
                transforms.RandomResizedCrop(size=(img_size,img_size), scale=(crop_scale_lower, 1.0), ratio=(0.75, 1.3333333333333333)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(convert_to_float)
            ])
        print('Using dvm transform for train augmentation')
    
    elif target == 'cardiac':
        # [修改] 这是之前 'else' 块的逻辑，现在明确给 cardiac
        if augmentation_speedup:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.0)),
                # A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.2, 1.0)),
                A.Lambda(name='convert2tensor', image=convert_to_ts_01) # 适用于 cardiac
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomResizedCrop(size=img_size, scale=(0.2,1)),
                transforms.Lambda(convert_to_float)
            ])
        print('Using cardiac transform for train augmentation')

    elif target in ['celeba', 'skin_cancer', 'adoption']:
        # [新增分支] 适用于 [0, 255] 范围的标准 RGB 图像
        print(f'Using {target} transform (RGB + 0-1 Norm) for train augmentation')
        if augmentation_speedup:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.0)),
                # A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.2, 1.0)),
                # [修复] 使用 ToTensorV2 进行归一化和维度转换
                ToTensorV2() 
            ])
        else:
            # (假设 non-speedup 路径也使用 'cardiac' 的 torchvision 增强)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomResizedCrop(size=img_size, scale=(0.2,1)),
                transforms.Lambda(convert_to_float) # 假设 non-speedup 路径加载的已是 [0, 1]
            ])

    elif target == 'breast_cancer':
        # [新增分支] 针对 breast_cancer (灰度图) 的特定修复
        print('Using breast_cancer transform (L -> RGB + 0-1 Norm) for train augmentation')
        if augmentation_speedup:
            transform = A.Compose([
                # 空间增强 (可以安全地在 1 通道上运行)
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=45),
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.0)),
                # A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.2, 1.0)),
              
                # [修复] 移除 ColorJitter
                
                # [修复] 在最后进行转换
                A.ToRGB(p=1.0), # 1. 转换为 3 通道
                ToTensorV2()   # 2. 归一化 [0, 255] -> [0, 1] 并 permute
            ])
        else:
             # (假设 non-speedup 路径也使用 'cardiac' 的 torchvision 增强)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                # 移除 ColorJitter
                transforms.RandomResizedCrop(size=img_size, scale=(0.2,1)),
                
                # [修复] 添加 Grayscale-to-RGB
                transforms.Grayscale(num_output_channels=3), 
                transforms.Lambda(convert_to_float)
            ])
            
    else:
        # [新增] 捕获未定义的
        raise ValueError(f"No augmentations defined in grab_image_augmentations for target: {target}")

    return transform



def grab_soft_eval_image_augmentations(img_size: int) -> transforms.Compose:
  """
  Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
  """
  transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.RandomResizedCrop(size=img_size, scale=(0.8,1)),
    # transforms.Lambda(lambda x: x.float())
    transforms.Lambda(lambda x: x/255.0)
  ])
  return transform

def grab_hard_eval_image_augmentations(img_size: int, target: str) -> transforms.Compose:
  """
  Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
  """
  if target.lower() == 'dvm_origin':
    transform = transforms.Compose([
      transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
      transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.Resize(size=(img_size,img_size)),
      # transforms.Lambda(lambda x : x.float())
      transforms.Lambda(lambda x: x/255.0)
    ])
  else:
    transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(45),
      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
      transforms.RandomResizedCrop(size=img_size, scale=(0.6,1)),
      # transforms.Lambda(lambda x: x.float())
      transforms.Lambda(lambda x: x/255.0)
    ])
  return transform

# def grab_image_augmentations(img_size: int, target: str, crop_scale_lower: float = 0.08) -> transforms.Compose:
#   """
#   Defines augmentations to be used with images during contrastive training and creates Compose.
#   """
#   if target.lower() == 'dvm':
#     transform = transforms.Compose([
#       transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
#       transforms.RandomGrayscale(p=0.2),
#       transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
#       transforms.RandomResizedCrop(size=(img_size,img_size), scale=(crop_scale_lower, 1.0), ratio=(0.75, 1.3333333333333333)),
#       transforms.RandomHorizontalFlip(p=0.5),
#       #transforms.Resize(size=(img_size,img_size)),
#       transforms.Lambda(lambda x : x.float())
#     ])
#   else:
#     transform = transforms.Compose([
#       transforms.RandomHorizontalFlip(),
#       transforms.RandomRotation(45),
#       transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#       transforms.RandomResizedCrop(size=img_size, scale=(0.2,1)),
#       transforms.Lambda(lambda x: x.float())
#     ])
#   return transform

# def grab_soft_eval_image_augmentations(img_size: int) -> transforms.Compose:
#   """
#   Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
#   """
#   transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(20),
#     transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
#     transforms.RandomResizedCrop(size=img_size, scale=(0.8,1)),
#     transforms.Lambda(lambda x: x.float())
#   ])
#   return transform

# def grab_hard_eval_image_augmentations(img_size: int, target: str) -> transforms.Compose:
#   """
#   Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
#   """
#   if target.lower() == 'dvm':
#     transform = transforms.Compose([
#       transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
#       transforms.RandomGrayscale(p=0.2),
#       transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
#       transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
#       transforms.RandomHorizontalFlip(p=0.5),
#       transforms.Resize(size=(img_size,img_size)),
#       transforms.Lambda(lambda x : x.float())
#     ])
#   else:
#     transform = transforms.Compose([
#       transforms.RandomHorizontalFlip(),
#       transforms.RandomRotation(45),
#       transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#       transforms.RandomResizedCrop(size=img_size, scale=(0.6,1)),
#       transforms.Lambda(lambda x: x.float())
#     ])
#   return transform

def grab_wids(category: str):
  # boat
  wids_b = ['n02951358', 'n03447447', 'n04612504', 'n03344393', 'n03662601', 'n04273569'] 
  # domestic cat
  wids_c = ['n02123597', 'n02123159', 'n02123045', 'n02124075', 'n02123394']
  # domestic dog
  wids_d = ['n02102480', 'n02096585', 'n02093256', 'n02091831', 'n02086910', 'n02100735', 'n02102040', 'n02085936', 'n02097130', 'n02097047', 'n02106662', 'n02110958', 'n02097209', 'n02092002', 'n02107142', 'n02099712', 'n02093754', 'n02112018', 'n02105412', 'n02096437', 'n02105251', 'n02108089', 'n02108551', 'n02095889', 'n02113624', 'n02093428', 'n02088238', 'n02100877', 'n02099849', 'n02108422', 'n02098413', 'n02086240', 'n02107574', 'n02101556', 'n02099429', 'n02098105', 'n02087394', 'n02108000', 'n02106166', 'n02107683', 'n02091244', 'n02101388', 'n02111889', 'n02093647', 'n02102973', 'n02101006', 'n02109961', 'n02085782', 'n02091635', 'n02112706', 'n02090622', 'n02110063', 'n02113712', 'n02110341', 'n02086079', 'n02089973', 'n02112350', 'n02113799', 'n02105162', 'n02108915', 'n02104029', 'n02089867', 'n02098286', 'n02105505', 'n02110627', 'n02106550', 'n02105641', 'n02100583', 'n02090721', 'n02093859', 'n02113978', 'n02088466', 'n02095570', 'n02099267', 'n02099601', 'n02106030', 'n02112137', 'n02089078', 'n02092339', 'n02088632', 'n02102177', 'n02096051', 'n02096294', 'n02096177', 'n02093991', 'n02110185', 'n02111277', 'n02090379', 'n02111500', 'n02088364', 'n02088094', 'n02094114', 'n02104365', 'n02111129', 'n02109525', 'n02097658', 'n02113186', 'n02095314', 'n02113023', 'n02087046', 'n02094258', 'n02100236', 'n02097298', 'n02105855', 'n02085620', 'n02106382', 'n02091032', 'n02110806', 'n02086646', 'n02094433', 'n02091134', 'n02107312', 'n02107908', 'n02097474', 'n02091467', 'n02102318', 'n02105056', 'n02109047']

  if category == 'Boat':
    return wids_b
  elif category == 'DomesticCat':
    return wids_c
  elif category == 'DomesticDog':
    return wids_d
  else:
    raise ValueError('Category not recognized')

def grab_arg_from_checkpoint(args: str, arg_name: str):
  """
  Loads a lightning checkpoint and returns an argument saved in that checkpoints hyperparameters
  """
  if args.checkpoint:
    ckpt = torch.load(args.checkpoint)
    load_args = ckpt['hyper_parameters']
  else:
    load_args = args
  return load_args[arg_name]

def chkpt_contains_arg(ckpt_path: str, arg_name: str):
  """
  Checks if a checkpoint contains a given argument.
  """
  ckpt = torch.load(ckpt_path)
  return arg_name in ckpt['hyper_parameters']

def prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'train_similarity_matrix', 'val_similarity_matrix', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_tabular', 'data_val_tabular', 
    'data_val_eval_tabular', 'labels_val_eval_tabular', 
    'data_train_eval_tabular', 'labels_train_eval_tabular',
    'field_indices_tabular', 'field_lengths_tabular',
    'data_test_eval_tabular', 'labels_test_eval_tabular',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams['{}_short'.format(hp)] = hparams[hp]
      hparams[hp] = join(db, hparams[hp])

  return hparams

def re_prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'train_similarity_matrix', 'val_similarity_matrix', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_tabular', 'data_val_tabular', 
    'data_val_eval_tabular', 'labels_val_eval_tabular', 
    'data_train_eval_tabular', 'labels_train_eval_tabular',
    'field_indices_tabular', 'field_lengths_tabular',
    'data_test_eval_tabular', 'labels_test_eval_tabular',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams[hp] = join(db, hparams['{}_short'.format(hp)])

  return hparams

# def re_prepend_paths(hparams):
#     print("\n--- [DEBUG] 进入 re_prepend_paths 函数 (最终版) ---")
    
#     if 'data_db' not in hparams or hparams.data_db is None:
#         raise ValueError("配置中缺少 'data_db'，无法修正路径。")
        
#     db = hparams.data_db
#     print(f"[*] 使用的基础数据路径 (db): {db}")

#     # =====================================================================
#     #  ▼▼▼ 核心修复：我们不再使用模糊的 'in' 匹配 ▼▼▼
#     #  而是定义一个明确的、需要处理的路径键列表
#     # =====================================================================
#     path_keys_to_process = [
#         'data_base',
#         'data_orig', 
#         'labels_train', 
#         'labels_val', 
#         'data_train_tabular', 
#         'data_val_tabular', 
#         'data_test_eval_tabular', 
#         'data_train_imaging', 
#         'data_val_imaging', 
#         'data_test_eval_imaging', 
#         'data_train_eval_imaging', 
#         'labels_train_eval_imaging', 
#         'labels_test_eval_imaging'
#     ]
    
#     print(f"[*] 将要检查和修正以下明确指定的路径键: {path_keys_to_process}")

#     # 只遍历我们明确定义的路径键列表
#     for hp in path_keys_to_process:
#         # 首先检查这个键是否存在于配置中
#         if hp not in hparams:
#             print(f"\n -> 警告: 在配置中未找到键 '{hp}'，跳过。")
#             continue

#         short_key = f'{hp}_short'
#         print(f"\n -> 正在处理长路径键: '{hp}'")
#         print(f"    - 正在查找对应的短路径键: '{short_key}'")
        
#         short_value = hparams.get(short_key)

#         if short_value is None:
#             print(f"\n❌ 致命错误: 在配置中找不到键 '{short_key}' 或其值为 None。")
#             print(f"   请检查您的 update_checkpoint.py 脚本，确保已经为 '{hp}' 添加了 '{short_key}'。")
#             raise KeyError(f"配置中缺失或为空的关键键: {short_key}")

#         print(f"    - 找到 '{short_key}' 的值: '{short_value}'")
#         hparams[hp] = join(db, short_value)
#         print(f"    - 成功拼接新路径: {hparams[hp]}")
        
#     print("--- [DEBUG] re_prepend_paths 函数执行完毕 ---\n")
#     return hparams

def cos_sim_collate(data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
  """
  Collate function to use when cosine similarity of embeddings is relevant. Takes the embeddings returned by the dataset and calculates the cosine similarity matrix for them.
  """
  view_1, view_2, labels, embeddings, thresholds = zip(*data)
  view_1 = torch.stack(view_1)
  view_2 = torch.stack(view_2)
  labels = torch.tensor(labels)
  threshold = thresholds[0]

  cos = torch.nn.CosineSimilarity(dim=0)
  cos_sim_matrix = torch.zeros((len(embeddings),len(embeddings)))
  for i in range(len(embeddings)):
      for j in range(i,len(embeddings)):
          val = cos(embeddings[i],embeddings[j]).item()
          cos_sim_matrix[i,j] = val
          cos_sim_matrix[j,i] = val

  if threshold:
    cos_sim_matrix = torch.threshold(cos_sim_matrix,threshold,0)

  return view_1, view_2, labels, cos_sim_matrix

def calc_logits_labels(out0, out1, temperature=0.1):
  out0 = nn.functional.normalize(out0, dim=1)
  out1 = nn.functional.normalize(out1, dim=1)

  logits = torch.matmul(out0, out1.T) / temperature
  labels = torch.arange(len(out0), device=out0.device)

  return logits, labels