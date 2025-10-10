import torch
from omegaconf import OmegaConf

# --- 需要你修改的配置 ---
# --- 需要你修改的配置 ---
CHECKPOINT_PATH = "/data0/jiazy/tab-image-bench/MMCL2/checkpoint_last_epoch_499_updated.ckpt"
NEW_CHECKPOINT_PATH = "/data0/jiazy/tab-image-bench/MMCL2/checkpoint_last_epoch_499_final.ckpt" 

# 将所有需要添加或更新的键值对放在这个字典里
KEYS_TO_ADD_OR_UPDATE = {
    "task": "classification",
    "data_base_short": "petfinder_adoptionprediction",
    "data_orig_short": "train_with_sentiment_clean.csv", # 这个可能不在那个目录，但保持原样
    "generate_embeddings_short": "",

    # =====================================================================
    #  ▼▼▼ 核心修改区域：为所有文件名加上 'petfinder_adoptionprediction/' 前缀 ▼▼▼
    # =====================================================================
    
    # --- 图像数据 ---
    "data_train_eval_imaging_short": "petfinder_adoptionprediction/images_train.pt",
    "data_val_eval_imaging_short":   "petfinder_adoptionprediction/images_valid.pt",
    "data_test_eval_imaging_short":  "petfinder_adoptionprediction/images_test.pt",
    
    # --- 表格数据 ---
    "data_train_eval_tabular_short": "petfinder_adoptionprediction/features_train.pt",
    "data_val_eval_tabular_short":   "petfinder_adoptionprediction/features_valid.pt",
    "data_test_eval_tabular_short":  "petfinder_adoptionprediction/features_test.pt",

    # --- 标签数据 ---
    "labels_train_eval_imaging_short": "petfinder_adoptionprediction/labels_train.pt",
    "labels_val_eval_imaging_short":   "petfinder_adoptionprediction/labels_valid.pt",
    "labels_test_eval_imaging_short":  "petfinder_adoptionprediction/labels_test.pt",

    # --- 其他可能的路径键 (以防万一) ---
    "labels_train_short": "petfinder_adoptionprediction/labels_train.pt",
    "labels_val_short":   "petfinder_adoptionprediction/labels_valid.pt",
    "data_train_tabular_short": "petfinder_adoptionprediction/features_train.pt",
    "data_val_tabular_short":   "petfinder_adoptionprediction/features_valid.pt",
    "data_train_imaging_short": "petfinder_adoptionprediction/images_train.pt",
    "data_val_imaging_short":   "petfinder_adoptionprediction/images_valid.pt"
}
# --- 修改结束 ---

print(f"[*] -----------------------------------------------------------------")
print(f"[*] 正在加载 checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))

if 'hyper_parameters' not in checkpoint:
    raise KeyError("在 checkpoint 中未找到 'hyper_parameters'。")

hparams = checkpoint['hyper_parameters']

print("[*] 临时解锁 hparams 的 'struct' 模式...")
OmegaConf.set_struct(hparams, False)

print("[*] 正在批量添加/更新以下配置项:")
for key, value in KEYS_TO_ADD_OR_UPDATE.items():
    print(f"  - {key}: {value}")
    hparams[key] = value

print("[*] 重新锁定 hparams 的 'struct' 模式...")
OmegaConf.set_struct(hparams, True)

checkpoint['hyper_parameters'] = hparams

torch.save(checkpoint, NEW_CHECKPOINT_PATH)
print(f"\n[成功] 新的最终 checkpoint 已保存到: {NEW_CHECKPOINT_PATH}")
print("[*] -----------------------------------------------------------------")