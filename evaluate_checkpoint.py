import os
import torch
from omegaconf import OmegaConf

# 导入您项目中的相关模块
from trainers.test import test
from utils.utils import re_prepend_paths

# 导入并禁用 wandb
import wandb
wandb.init(mode="disabled")
from pytorch_lightning.loggers import WandbLogger

# --- 1. 配置区域：请在这里填写您的路径 ---

# 指定您要评测的、已经训练完成的 checkpoint 文件路径
# !! 请确保使用的是我们之前用 update_checkpoint.py 生成的 _updated.ckpt 或 _final.ckpt 文件 !!
CHECKPOINT_PATH = "/data0/jiazy/tab-image-bench/MMCL2/checkpoint_last_epoch_499_final.ckpt"

# 指定数据集的根目录
BASE_DATA_DIR = "/data1/jiazy/tab_image_bench/PetFinder_datasets/dataset"

# --- 配置结束 ---


def evaluate_from_checkpoint():
    """
    一个专门用于加载 checkpoint 并进行测试的函数。
    """
    print(f"===== 开始从 Checkpoint 进行评测 =====")

    # --- 步骤 1: 加载 Checkpoint 和配置 ---
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 错误: Checkpoint 文件未找到: {CHECKPOINT_PATH}")
        return

    print(f"[*] 正在加载 Checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    if 'hyper_parameters' not in ckpt:
        print("❌ 错误: 在 checkpoint 中未找到 'hyper_parameters'。")
        return
        
    args = OmegaConf.create(ckpt['hyper_parameters'])
    print("[*] 已成功从 Checkpoint 中加载配置 (hparams)。")

    # --- 步骤 2: 临时解锁配置结构，并更新评测所需参数 ---
    print("[*] 临时解锁配置 'struct' 模式以进行修改...")
    OmegaConf.set_struct(args, False)

    # 现在可以安全地添加或修改配置了
    args.checkpoint = CHECKPOINT_PATH
    args.resume_training = False
    args.pretrain = False
    args.test = True
    args.data_db = BASE_DATA_DIR # <-- 现在这一行可以成功执行了

    # (可选) 修改完成后，建议重新锁定结构
    OmegaConf.set_struct(args, True)
    print("[*] 配置已更新并重新锁定。")
    
    # --- 步骤 3: 修正配置文件中的所有数据路径 ---
    print("[*] 正在调用 re_prepend_paths 来修正文件路径...")
    args = re_prepend_paths(args)
    print("[*] 文件路径修正完成。")
    
    print("\n--- 检查修正后的关键路径 ---")
    print(f"  - 图像测试集: {args.get('data_test_eval_imaging', '未找到')}")
    print(f"  - 标签测试集: {args.get('labels_test_eval_imaging', '未找到')}")
    print("----------------------------\n")

    # --- 步骤 4: 创建离线 Logger 并调用 test 函数 ---
    wandb_logger = WandbLogger(project="evaluation", offline=True)
    
    print("[*] 所有准备工作完成，正在调用 test() 函数...")
    test(args, wandb_logger, model=None)
    
    print("\n🎉 评测流程已完成！ 🎉")


if __name__ == "__main__":
    evaluate_from_checkpoint()