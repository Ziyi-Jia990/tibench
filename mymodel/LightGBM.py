import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
import sys
import torch
import os

# --- Hydra 导入 ---
import hydra
from omegaconf import DictConfig, OmegaConf

def load_and_preprocess_data(cfg: DictConfig):
    """
    根据 config 对象加载和预处理数据。
    """
    target_name = cfg.get('target', 'unknown')
    task = cfg.get('task', 'classification')
    print(f"--- 1. 正在为 Target: '{target_name}' (任务: {task}) 加载数据 ---")

    drop_cols = []
    
    # ==================================================================
    # == 条件数据加载逻辑 ==
    # ==================================================================
    
    if target_name == 'dvm':
        print("检测到 'dvm' target，正在从 .csv 和 .pt 加载数据...")
        try:
            # --- 关键修复: 添加 header=None ---
            X_train = pd.read_csv(cfg.data_train_tabular, header=None)
            
            # (y_train 的加载逻辑不变)
            y_train_tensor = torch.load(cfg.labels_train, weights_only=False) 
            if isinstance(y_train_tensor, torch.Tensor):
                y_train = pd.Series(y_train_tensor.numpy())
            else:
                y_train = pd.Series(y_train_tensor)

            # --- 关键修复: 验证集同样添加 header=None ---
            X_test = pd.read_csv(cfg.data_val_tabular, header=None)
            
            # (y_test 的加载逻辑不变)
            y_test_tensor = torch.load(cfg.labels_val, weights_only=False)
            if isinstance(y_test_tensor, torch.Tensor):
                y_test = pd.Series(y_test_tensor.numpy())
            else:
                y_test = pd.Series(y_test_tensor)
            
            print("DVM 数据加载成功。")

            # dvm 没有需要丢弃的列
            drop_cols = [] 
            
        except FileNotFoundError as e:
            print(f"错误: 找不到DVM所需的文件 {e.filename}。")
            sys.exit(1)
        except KeyError as e:
            print(f"错误: 配置文件中缺少DVM所需的键: {e}")
            sys.exit(1)
            

    elif target_name == 'adoption':
        # --- 修复后的 Adoption 逻辑 ---
        print("检测到 'adoption' target, 正在从 .csv 加载数据...")
        try:
            # 1. 从 config 获取 adoption 特定的键
            target_col = 'AdoptionSpeed'
            train_df = pd.read_csv(cfg.train_path)
            test_df = pd.read_csv(cfg.test_path)

            # 2. 从 .csv 中分离特征 (X) 和标签 (y)
            y_train = train_df[target_col]
            X_train = train_df.drop(columns=[target_col])
            
            y_test = test_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            
            print("adoption 数据加载成功。")

            # 3. 指定要丢弃的列 (注意：AdoptionSpeed 已经被分离为 y，不再 X 中)
            drop_cols = ['RescuerID', 'Description', 'PetID'] 
            
        except FileNotFoundError as e:
            print(f"错误: 找不到adoption所需的文件 {e.filename}。")
            sys.exit(1)
        except KeyError as e:
            # 如果 config 中没有 target_col, train_path, test_path，会触发此错误
            print(f"错误: 配置文件中缺少adoption所需的键: {e}")
            sys.exit(1)
            
    else:
        print(f"错误: 未知的 target '{target_name}'。没有定义数据加载逻辑。")
        sys.exit(1)

    # ==================================================================
    # == 通用预处理逻辑 (此部分您的代码是正确的) ==
    # ==================================================================

    print(f"原始训练集大小: {X_train.shape}")
    print(f"原始测试集(验证集)大小: {X_test.shape}")

    # --- 1. 丢弃列 ---
    valid_drop_cols = [col for col in drop_cols if col in X_train.columns]
    if valid_drop_cols:
        print(f"正在丢弃 {len(valid_drop_cols)} 列: {valid_drop_cols}")
        X_train = X_train.drop(columns=valid_drop_cols)
        valid_drop_cols_test = [col for col in drop_cols if col in X_test.columns]
        X_test = X_test.drop(columns=valid_drop_cols_test)
    else:
        print("没有指定或找到需要丢弃的列。")

    # --- 2. 对齐列 ---
    print("正在对齐训练集和测试集的列...")
    X_test = X_test[X_train.columns]

    # --- 3. 转换分类特征 ---
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"自动识别出的分类特征 ({len(categorical_features)}): {categorical_features}")

    for col in categorical_features:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories, ordered=False)
        
    print("分类特征转换完成。")
    print("-" * 30)

    # --- 4. 确定问题类型和评估指标 ---
    if task == 'classification':
        num_classes = cfg.get('num_classes', len(np.unique(y_train)))
        
        if num_classes == 2:
            print(f"检测到二分类问题 (num_classes={num_classes})。")
            problem_type = 'binary'
            objective = 'binary'
            num_class_param = {}
            scoring_metric = 'roc_auc'
        else:
            print(f"检测到多分类问题 (num_classes={num_classes})。")
            problem_type = 'multiclass'
            objective = 'multiclass'
            num_class_param = {'num_class': num_classes}
            scoring_metric = 'accuracy'
    
    elif task == 'regression':
        print("检测到回归问题。")
        problem_type = 'regression'
        objective = 'regression_l2'
        num_class_param = {}
        scoring_metric = 'neg_root_mean_squared_error'
    
    else:
        print(f"错误: 不支持的任务类型 '{task}'。")
        sys.exit(1)

    print(f"LGBM Objective: {objective}, GridSearchCV Scoring: {scoring_metric}")
    print("-" * 30)

    return X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric

# ... get_model_and_grid 和 run_experiment 函数 (它们是正确的，无需修改) ...

def get_model_and_grid(problem_type, objective, num_class_param, seed):
    """
    根据问题类型获取LGBM模型和参数网格。
    """
    if problem_type in ['binary', 'multiclass']:
        model = lgb.LGBMClassifier(
            objective=objective,
            **num_class_param,
            random_state=seed,
            n_jobs=1 # 限制LGBM的线程数，避免与GridSearchCV冲突
        )
    elif problem_type == 'regression':
        model = lgb.LGBMRegressor(
            objective=objective,
            random_state=seed,
            n_jobs=1
        )
    
    param_grid = {
        'num_leaves': [31, 127],
        'learning_rate': [0.01, 0.1],
        'min_child_samples': [20, 50, 100],
        'min_sum_hessian_in_leaf': [1e-3, 1e-2, 1e-1]
    }
    
    return model, param_grid


def run_experiment(X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric, seed):
    """
    使用给定的随机种子运行一次模型训练和评估。
    """
    print(f"\n{'='*25} ---------------- 随机种子: {seed} ---------------- {'='*25}")
    
    model, param_grid = get_model_and_grid(problem_type, objective, num_class_param, seed)

    print(f"开始进行网格搜索 (评分指标: {scoring_metric})...")
    
    if problem_type == 'regression':
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv_splitter,
        n_jobs=-1, # GridSearchCV 使用所有核心
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("网格搜索完成！")
    print(f"找到的最佳超参数: {grid_search.best_params_}")
    print(f"在交叉验证中的最佳 {scoring_metric}: {grid_search.best_score_:.4f}")
    print("-" * 30)

    print("使用最佳模型在测试集(验证集)上进行最终评估...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    if problem_type in ['binary', 'multiclass']:
        y_pred_proba = best_model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        if problem_type == 'binary':
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        
        result_line = f"acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}"
    
    elif problem_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        result_line = f"rmse:{rmse:.4f}"

    print("评估结果:")
    print(result_line)
    
    return result_line, grid_search.best_params_

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    
    print("--- 最终配置: ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------")
    print(f"Hydra 工作目录: {os.getcwd()}")
    print("--------------------")

    # 1. 加载和预处理数据
    X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric = load_and_preprocess_data(cfg)

    # 允许 config.yaml 或命令行覆盖 'output_file_name'
    output_filename = 'lgb_results.txt'
    seeds = list(cfg.get('seeds', [2023, 2025, 2026])) 
    # seeds=[2022, 2023, 2024]
    
    # 3. 打开文件准备写入结果
    print(f"\n准备将结果写入到文件: {output_filename}")
    with open(output_filename, 'w') as f:
        f.write("--- 最终配置 ---\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("-" * 30 + "\n\n")

        # 4. 循环遍历所有随机种子
        for seed in seeds:
            result_line, best_params = run_experiment(
                X_train, y_train, X_test, y_test,
                problem_type, objective, num_class_param, scoring_metric,
                seed
            )
            
            # 5. 写入结果
            print(f"正在将种子 {seed} 的结果写入到 {output_filename}...")
            f.write(f"seed:{seed}\n")
            f.write(f"best_params: {best_params}\n")
            f.write(result_line + "\n\n")

    print(f"\n所有任务完成！结果已全部保存在 '{os.getcwd()}/{output_filename}' 中。")

if __name__ == "__main__":
    main()