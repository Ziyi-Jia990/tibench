# eval_tabpfn.py
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, log_loss

from tabpfn import TabPFNClassifier

# =========================
# 路径与目标列
# =========================
TRAIN_PATH = '/data1/jiazy/tab_image_bench/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_train.csv'
TEST_PATH  = '/data1/jiazy/tab_image_bench/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_test.csv'
TARGET_NAME = 'AdoptionSpeed'

# =========================
# 工具函数（保留原预处理逻辑）
# =========================
ID_LIKE_COLS = ['PetID', 'RescuerID']
TEXT_LENGTH_DROP_THRESHOLD = 30
HIGH_CARDINALITY_THRESHOLD = 200

def drop_huge_text_and_high_cardinality(df: pd.DataFrame, y_col: str):
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
        print(f"[INFO] Drop columns: {drop_cols}")
    return df.drop(columns=drop_cols)

def load_data(train_csv: str, test_csv: str, target_name: str):
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    if target_name not in train_df.columns:
        raise ValueError(f"训练集缺少目标列 {target_name}")

    train_df = drop_huge_text_and_high_cardinality(train_df, target_name)
    test_df  = test_df[[c for c in test_df.columns if c in train_df.columns or c == target_name]]

    y = train_df[target_name].values
    X = train_df.drop(columns=[target_name])

    common_cols = [c for c in X.columns if c in test_df.columns]
    X = X[common_cols]
    X_test = test_df[common_cols].copy()

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
        X_test[c] = X_test[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].fillna("missing")
        X_test[c] = X_test[c].fillna("missing")

    return X, y, X_test, num_cols, cat_cols

def build_preprocess(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', sparse=True), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=1.0)

def stratified_subsample_indices(y, sample_size, seed):
    if len(y) <= sample_size:
        return np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
    idx_all = np.arange(len(y))
    for sub_idx, _ in sss.split(idx_all, y):
        return sub_idx

def evaluate_metrics(y_true, y_pred, y_proba=None):
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
    }
    if y_proba is not None:
        try:
            res["log_loss"] = log_loss(y_true, y_proba, labels=np.arange(y_proba.shape[1]))
        except Exception:
            pass
    return res

# =========================
# 主流程
# =========================
def main(args):
    # 读取与预处理
    X, y, X_test, num_cols, cat_cols = load_data(TRAIN_PATH, TEST_PATH, TARGET_NAME)

    # 采样子集
    sub_idx = stratified_subsample_indices(y, args.sample_size, args.seed)
    X_sub, y_sub = X.iloc[sub_idx], y[sub_idx]
    print(f"[INFO] Subsample with seed={args.seed}, size={len(sub_idx)}")

    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.2, random_state=args.seed, stratify=y_sub)

    preprocess = build_preprocess(num_cols, cat_cols)
    pipeline = Pipeline([("preprocess", preprocess)])
    pipeline.fit(X_train)

    X_train_np = pipeline.transform(X_train)
    X_val_np   = pipeline.transform(X_val)
    if hasattr(X_train_np, "toarray"):
        X_train_np = X_train_np.toarray()
        X_val_np   = X_val_np.toarray()

    # TabPFN 训练
    clf = TabPFNClassifier(N_ensemble_configurations=16, device='cuda')
    clf.fit(X_train_np, y_train)

    # 评估
    val_proba = clf.predict_proba(X_val_np)
    val_pred = np.argmax(val_proba, axis=1)
    metrics = evaluate_metrics(y_val, val_pred, val_proba)
    print("[RESULT] Validation metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="随机种子")
    parser.add_argument("--sample_size", type=int, default=8500, help="子采样大小")
    args = parser.parse_args()
    main(args)
