# eval_tabpfn.py
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import torch # 新增

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, log_loss

from tabpfn import TabPFNClassifier
from tabpfn_extensions.many_class.many_class_classifier import ManyClassClassifier

# --- Hydra 导入 ---
import hydra
from omegaconf import DictConfig, OmegaConf

# =========================
# 工具函数（修改为从 cfg 读取）
# =========================

# 从 cfg 中读取固定的阈值，如果cfg中没有，则使用默认值
TEXT_LENGTH_DROP_THRESHOLD = 30
HIGH_CARDINALITY_THRESHOLD = 200
N_ENSEMBLE_CONFIGURATIONS = 16

def drop_huge_text_and_high_cardinality(df: pd.DataFrame, y_col: str, cfg: DictConfig):
    # 从 config 中安全地获取 id_like_cols
    ID_LIKE_COLS = []
    if cfg.target=='dvm':
        ID_LIKE_COLS=[]
    elif cfg.target=='adoption':
        ID_LIKE_COLS=['PetID', 'RescuerID', 'Description']
    
    drop_cols = []
    for col in df.columns:
        if col == y_col:
            continue
        if col in ID_LIKE_COLS:
            drop_cols.append(col)
            continue
        if df[col].dtype == 'object':
            sample = df[col].dropna().astype(str)
            if not sample.empty:
                mean_len = sample.map(len).mean()
                nunique = df[col].nunique(dropna=True)
                if mean_len > TEXT_LENGTH_DROP_THRESHOLD:
                    drop_cols.append(col)
                    continue
                if nunique > HIGH_CARDINALITY_THRESHOLD:
                    drop_cols.append(col)
                    continue
    if drop_cols:
        print(f"[INFO] 自动丢弃列: {drop_cols}")
    return df.drop(columns=drop_cols)

def load_data(cfg: DictConfig):
    """
    重构的数据加载器：根据 cfg.target 条件加载。
    """
    target = cfg.target
    print(f"[INFO] 正在加载 target: {target}")
    
    if target == 'adoption':
        try:
            target_col = 'AdoptionSpeed'
            # 1. 加载完整训练集
            train_df = pd.read_csv(cfg.data_train_tabular) # 假设路径键为 data_train_tabular
            if target_col not in train_df.columns:
                raise ValueError(f"训练集缺少目标列 {target_col}")
            train_df = drop_huge_text_and_high_cardinality(train_df, target_col, cfg)
            y_train_full = train_df[target_col].values
            X_train_full = train_df.drop(columns=[target_col])
            
            # 2. 加载完整测试集
            test_df = pd.read_csv(cfg.data_test_eval_tabular) # 必须使用 _test_eval_ 路径
            if target_col not in test_df.columns:
                raise ValueError(f"测试集缺少目标列 {target_col}")
            # 注意：测试集上的 drop_huge... 仅用于丢弃ID列，不拟合
            test_df = drop_huge_text_and_high_cardinality(test_df, target_col, cfg)
            y_test_full = test_df[target_col].values
            X_test_full = test_df.drop(columns=[target_col])

        except FileNotFoundError as e:
            print(f"错误: 找不到 adoption 所需的文件 {e.filename}。")
            sys.exit(1)
        except KeyError as e:
            print(f"错误: adoption 配置文件中缺少键: {e} (请确保有 data_train_tabular, data_test_eval_tabular, target_col)")
            sys.exit(1)

    elif 'dvm' in target: # 捕获 'dvm', 'dvm_all_server' 等
        try:
            # 1. 加载完整训练集 (DVM 没有表头)
            X_train_full = pd.read_csv(cfg.data_train_tabular, header=None)
            y_tensor_train = torch.load(cfg.labels_train, weights_only=False) 
            y_train_full = np.array(y_tensor_train) if not isinstance(y_tensor_train, torch.Tensor) else y_tensor_train.numpy()
            X_train_full = drop_huge_text_and_high_cardinality(X_train_full, None, cfg)

            # 2. 加载完整测试集 (来自 ..._test_eval_...)
            X_test_full = pd.read_csv(cfg.data_test_eval_tabular, header=None)
            y_tensor_test = torch.load(cfg.labels_test_eval_tabular, weights_only=False)
            y_test_full = np.array(y_tensor_test) if not isinstance(y_tensor_test, torch.Tensor) else y_tensor_test.numpy()
            X_test_full = drop_huge_text_and_high_cardinality(X_test_full, None, cfg)
            
        except FileNotFoundError as e:
            print(f"错误: 找不到 dvm 所需的文件 {e.filename}。")
            sys.exit(1)
        except KeyError as e:
            print(f"错误: dvm 配置文件中缺少键: {e} (请确保有 data_train_tabular, labels_train, data_test_eval_tabular, labels_test_eval_tabular)")
            sys.exit(1)
    
    else:
        raise ValueError(f"未知的 dataset.target: {target}")

    # --- 通用预处理 ---
    
    # 列对齐：确保测试集与训练集具有完全相同的列
    X_test_full = X_test_full[X_train_full.columns]

    num_cols = [c for c in X_train_full.columns if pd.api.types.is_numeric_dtype(X_train_full[c])]
    cat_cols = [c for c in X_train_full.columns if not pd.api.types.is_numeric_dtype(X_train_full[c])]

    print(f"[INFO] 数值列: {num_cols}")
    print(f"[INFO] 分类列: {cat_cols}")
    print("[INFO] 跳过缺失值填充。")

    return X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols

def build_preprocess(num_cols, cat_cols):
    """
    修改：确保输出为 TabPFN 需要的密集 (dense) 矩阵。
    """
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        # 关键修复：sparse=True -> sparse_output=False
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)) 
    
    # 关键修复：sparse_threshold=1.0 -> 0.0
    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.0)

def stratified_subsample_indices(y, sample_size, seed):
    """
    从 y 中获取用于采样的索引 (您的代码已正确)
    """
    # 确保 sample_size 是整数
    sample_size = int(sample_size)
    if len(y) <= sample_size:
        return np.arange(len(y))
    
    # 确保 y 中至少有2个类别，或者足够的样本
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or (counts < 2).any():
        print("[WARNING] 类别太少或样本不足，无法进行分层采样，退回到随机采样。")
        np.random.seed(seed)
        return np.random.choice(np.arange(len(y)), sample_size, replace=False)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
    idx_all = np.arange(len(y))
    try:
        for sub_idx, _ in sss.split(idx_all, y):
            return sub_idx
    except ValueError as e:
        print(f"[WARNING] 分层采样失败 ({e})，退回到随机采样。")
        np.random.seed(seed)
        return np.random.choice(idx_all, sample_size, replace=False)

def evaluate_metrics(y_true, y_pred, y_proba=None):
    """
    (您的代码已正确)
    """
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
    }
    if y_proba is not None:
        try:
            # 确保 y_proba 的列数与类别数一致
            n_classes = y_proba.shape[1]
            # 确保 y_true 中的标签在 [0, n_classes-1] 范围内
            if y_true.max() >= n_classes:
                 print(f"[WARNING] y_true 包含标签 {y_true.max()}，但 y_proba 只有 {n_classes} 列。LogLoss 可能不准确。")
            
            res["log_loss"] = log_loss(y_true, y_proba, labels=np.arange(n_classes))
        except Exception as e:
            print(f"[WARNING] 无法计算 LogLoss: {e}")
            pass
    return res

# =========================
# 主流程 (Hydra)
# =========================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    
    print("--- 最终配置: ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------")
    print(f"Hydra 工作目录: {os.getcwd()}")
    print("--------------------")
    
    # 从 config 中获取参数 (假设在 configs/config.yaml 中定义了 'run' 模块)
    try:
        seed = cfg.seed
        sample_size = cfg.get('sample_size', 8500)
    except KeyError as e:
        # 修复错误提示，不再提 'run' 模块
        print(f"错误: 您的 configs/config.yaml 中缺少顶层键: {e}") 
        print("请确保 configs/config.yaml 包含 'seed' 和 'sample_size'")
        sys.exit(1)
    

    # 读取与预处理 (使用 cfg)
    X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols = load_data(cfg)

    # 3. 采样子集 (仅用于训练)
    sub_idx = stratified_subsample_indices(y_train_full, sample_size, seed)
    X_train_sampled = X_train_full.iloc[sub_idx]
    y_train_sampled = y_train_full[sub_idx]
    print(f"[INFO] 从训练集中采样完成，种子={seed}, 实际大小={len(y_train_sampled)}")

    # 4. 预处理
    preprocess = build_preprocess(num_cols, cat_cols)
    
    # *仅*在训练样本上拟合 (fit_transform)
    X_train_np = preprocess.fit_transform(X_train_sampled)
    
    print("[INFO] 正在转换 (transform) 完整测试集...")
    # *仅*在测试集上转换 (transform)
    X_test_np = preprocess.transform(X_test_full)
    
    print(f"[INFO] 预处理后训练集形态: {X_train_np.shape}")
    print(f"[INFO] 预处理后测试集形态: {X_test_np.shape}")

    # 5. TabPFN 训练
    # 假设 TabPFN 超参数在 config.yaml 的 'tabpfn' 模块下
    n_ensemble = N_ENSEMBLE_CONFIGURATIONS
    device = 'cuda'
    
    if cfg.target=='adoption':
        clf = TabPFNClassifier(n_estimators=n_ensemble, device=device)
    elif cfg.target=='dvm':
        base_clf = TabPFNClassifier(n_estimators=n_ensemble, device=device)
        clf = ManyClassClassifier(
            estimator=base_clf,      # 将你的 TabPFN 实例作为基础估计器
            alphabet_size=10,        # 关键参数：告诉包装器，你的基础模型最多只能处理 10 个类
            random_state=seed,         # 确保结果可复现
            verbose=1                # 可以打印出 codebook 的统计信息，方便调试
        )
    
    print("Fitting ManyClassClassifier wrapper...")
    clf.fit(X_train_np, y_train_sampled)
    print("Fit complete.")

    # 6. 评估 (在完整的测试集上)
    print("[INFO] 正在评估完整测试集...")
    print(f"DEBUG: Shape of X_test_np: {X_test_np.shape}")
    print(f"DEBUG: Shape of X_train_np (for comparison): {X_train_np.shape}")
    if cfg.target=='adoption':
        X_test_sampled = X_test_np
        y_test_sampled = y_test_full  # 默认使用全集
    elif cfg.target=='dvm':
        TARGET_TEST_SIZE = 2000
        X_test_sampled = X_test_np
        y_test_sampled = y_test_full  # 默认使用全集

        # 检查测试集是否大于目标大小，如果大，则进行分层采样
        if len(X_test_np) > TARGET_TEST_SIZE:
            print(f"WARN: 完整测试集 ({len(X_test_np)}) 样本过多。")
            print(f"WARN: 正在进行分层采样，缩减至 {TARGET_TEST_SIZE} 个样本...")
            
            try:
                # 使用 train_test_split 来实现分层采样
                # 我们只保留 `train_size` 部分，所以将其设为我们的目标大小
                X_test_sampled, _, y_test_sampled, _ = train_test_split(
                    X_test_np, 
                    y_test_full,  # 必须提供标签来进行分层
                    train_size=TARGET_TEST_SIZE, 
                    stratify=y_test_full,  # <-- 关键：按标签比例进行采样
                    random_state=seed      # 保证采样结果可复现
                )
                print(f"DEBUG: 采样后测试集特征形态: {X_test_sampled.shape}")
                print(f"DEBUG: 采样后测试集标签形态: {y_test_sampled.shape}")
                
            except Exception as e:
                print(f"ERROR: 分层采样失败: {e}")
                print("ERROR: 请确保 'y_test_np' 变量已正确加载。")
                # 你可以在这里选择是退出还是继续使用完整数据集（如果内存允许）
                raise e

        else:
            print(f"DEBUG: 测试集 ({len(X_test_np)}) 小于目标大小，将使用完整测试集。")


    test_proba = clf.predict_proba(X_test_sampled)
    test_pred = np.argmax(test_proba, axis=1)

    print(f"DEBUG: Final test_proba shape: {test_proba.shape}")
    print(f"DEBUG: Final test_pred shape: {test_pred.shape}")
            
    # 使用 y_test_full 进行评估
    metrics = evaluate_metrics(y_test_sampled, test_pred, test_proba)
    
    print("[RESULT] 完整测试集指标:")
    print(json.dumps(metrics, indent=2))

    # 7. 保存结果
    output_file = "tabpfn_results.json"
    with open(output_file, 'a') as f:
        f.write(f"seed: {seed}\n") 
        json.dump(metrics, f, indent=2)
    print(f"[INFO] 结果已保存到 {os.getcwd()}/{output_file}")


if __name__ == "__main__":
    main()