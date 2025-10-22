import os
import pandas as pd
import torch

# --- 配置区域 ---
# 指向您之前预处理好的、包含所有特征的训练集CSV文件
# 注意：我们用的是最开始生成的 dataset_train.csv，而不是 features_train.pt
DATA_DIRECTORY = "/data1/jiazy/tab_image_bench/PetFinder_datasets/dataset/petfinder_adoptionprediction"
TRAIN_CSV_PATH = os.path.join(DATA_DIRECTORY, "dataset_train.csv")

# 定义输出文件名
OUTPUT_FILENAME = "tabular_lengths_petfinder.pt"
OUTPUT_PATH = os.path.join(DATA_DIRECTORY, OUTPUT_FILENAME)

# --- 脚本逻辑 ---

def create_field_lengths():
    """
    读取表格数据，计算每个字段的长度（类别特征的唯一值数量，数值特征为1），
    并保存为一个 .pt 文件。
    """
    print(f"[*] 正在读取训练集CSV文件: {TRAIN_CSV_PATH}")
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"❌ 错误: 找不到文件 {TRAIN_CSV_PATH}。")
        return

    df = pd.read_csv(TRAIN_CSV_PATH)
    
    # 从数据中移除标签和ID/文本列，只留下特征
    features_df = df.drop(columns=['AdoptionSpeed', 'PetID', 'Description'], errors='ignore')
    
    print("[*] 正在计算每个特征字段的长度...")
    field_lengths = []
    
    # 遍历所有特征列
    for col in features_df.columns:
        # 判断是类别特征 (object/string)还是数值特征
        if features_df[col].dtype == 'object':
            # 类别特征：长度是唯一值的数量
            num_unique = features_df[col].nunique()
            field_lengths.append(num_unique)
            print(f"  - 类别特征 '{col}': {num_unique} 个唯一值")
        else:
            # 数值特征：长度视为 1
            field_lengths.append(1)
            print(f"  - 数值特征 '{col}': 长度为 1")
            
    # 将列表转换为 PyTorch 张量
    lengths_tensor = torch.tensor(field_lengths, dtype=torch.long)
    
    print(f"\n[*] 计算完成的字段长度列表: {field_lengths}")
    print(f"[*] 最终张量的形状: {lengths_tensor.shape}")
    
    # 保存张量
    torch.save(lengths_tensor, OUTPUT_PATH)
    print(f"\n✅ 成功将字段长度信息保存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_field_lengths()