import torch

# --- 配置区域 ---
# 只需将此路径替换为您想要检查的 checkpoint 文件路径即可
CHECKPOINT_PATH = "/data1/jiazy/tab-image-bench/MMCL/checkpoint_last_epoch_499.ckpt"
# --- 结束配置 ---

def read_hparams_from_checkpoint(path: str):
    """
    加载一个 PyTorch Lightning checkpoint 并打印出指定的超参数。
    """
    print(f"[*] 正在加载 checkpoint: {path}")

    # 使用 map_location='cpu' 确保即使在没有GPU的机器上也能成功加载
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 -> {path}")
        return

    # 超参数通常存储在 'hyper_parameters' 键下
    if 'hyper_parameters' not in checkpoint:
        print("❌ 错误: 在此 checkpoint 中未找到 'hyper_parameters'。")
        return

    hparams = checkpoint['hyper_parameters']
    
    print("\n--- 成功读取超参数 ---")

    # 1. 读取 batch_size
    # 它通常是一个顶层键
    try:
        batch_size = hparams['batch_size']
        print(f"Batch Size: {batch_size}")
    except KeyError:
        print("- 未找到 'batch_size' 键。")

    # 2. 读取 optimizer.lr
    # 它通常嵌套在 'optimizer' 字典中
    try:
        # 访问嵌套的键
        learning_rate = hparams['optimizer']['lr']
        print(f"Optimizer Learning Rate: {learning_rate}")
    except KeyError:
        print("- 未找到 'optimizer.lr' 键。")

    # (可选) 如果你想查看所有保存的超参数，可以取消下面这行代码的注释
    # print("\n--- 所有保存的超参数 ---")
    # print(hparams)


if __name__ == "__main__":
    read_hparams_from_checkpoint(CHECKPOINT_PATH)