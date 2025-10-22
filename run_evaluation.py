import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb

# 确保可以正确导入您项目中的模块
from trainers.evaluate import evaluate

# 初始化 wandb 为离线模式
wandb.init(mode="disabled")

@hydra.main(config_path='./configs', config_name='config_evaluate', version_base=None)
def main_evaluation(args: DictConfig):
    """
    最终解决方案：使用 data_base_path 为所有从 checkpoint 加载的相对路径添加前缀。
    这个方法简单、直接，并且能与项目现有代码（如 re_prepend_paths）和谐共存。
    """
    print("--- [启动特征评估流程 (data_base_path 方案)] ---")

    # 1. 加载 Checkpoint 和其内部的原始配置
    checkpoint_path = args.checkpoint
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"错误：必须提供一个有效的 checkpoint 路径。未找到文件: '{checkpoint_path}'")

    print(f"[*] 使用的 Checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    original_hparams = OmegaConf.create(ckpt['hyper_parameters'])
    print("[*] 已成功从 Checkpoint 加载原始训练配置。")

    # 2. 合并配置，命令行或YAML文件中的参数会覆盖 Checkpoint 中的值
    OmegaConf.set_struct(original_hparams, False)
    final_hparams = OmegaConf.merge(original_hparams, args)

    # 3. 【核心逻辑】修正所有数据路径
    data_base_path = final_hparams.get('data_base_path', None)
    if not data_base_path:
        raise ValueError("错误：必须通过命令行或 'config_evaluate.yaml' 提供 'data_base_path' 参数！")

    print(f"\n[*] 将使用基础路径为所有相关数据路径添加前缀: {data_base_path}")

    # 遍历所有配置项
    for key, value in final_hparams.items():
        # 如果值是字符串并且看起来像一个需要修正的相对路径 (这里以 'petfinder' 作为标识)
        if isinstance(value, str) and 'petfinder_adoptionprediction' in value:
            absolute_path = os.path.join(data_base_path, value)
            final_hparams[key] = absolute_path
            # print(f"    - 修正路径 {key}: {absolute_path}") # 如需调试可以取消此行注释

    print("[*] 所有数据路径修正完毕。")

    print("\n--- [最终评估配置] ---")
    print(OmegaConf.to_yaml(final_hparams))

    # 4. 创建 Logger 并调用 evaluate 函数
    wandb_logger = WandbLogger(project="evaluation-only", save_dir="evaluation_logs", offline=True)
    print("\n--- [正在调用 evaluate 函数] ---")
    evaluate(final_hparams, wandb_logger)
    print("\n--- [评估流程执行完毕] ---")


if __name__ == "__main__":
    main_evaluation()