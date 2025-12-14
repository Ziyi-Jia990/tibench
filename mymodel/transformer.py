import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import rtdl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error
import time
import os
import random
import sys
import json

# --- Hydra 导入 ---
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path # 

# --- 0. 配置与设置随机种子 ---
def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 自定义模型类 ---
class MyFTTransformer(nn.Module):
    def __init__(self, ft_transformer_module, d_token, d_out):
        super().__init__()
        self.ft_transformer = ft_transformer_module
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num, x_cat):
        x = self.ft_transformer(x_num, x_cat)
        x = self.head(x)
        # 回归任务可能需要 squeeze
        if x.shape[-1] == 1:
            return x.squeeze(-1)
        return x

# --- 1. 数据加载与预处理 (重构为函数) ---
def load_and_preprocess_data(cfg: DictConfig, batch_size:int):
    """
    根据 config 对象加载和预处理数据。
    """
    dataset_name = cfg.target
    print(f"--- 1. 正在为数据集: '{dataset_name}' 加载数据 ---")

    X_train_num, X_train_cat, y_train = None, None, None
    X_val_num, X_val_cat, y_val = None, None, None
    X_test_num, X_test_cat, y_test = None, None, None
    numerical_features, categorical_features = [], []
    cat_cardinalities = []

    # ==================================================================
    # == Adoption 数据集逻辑
    # ==================================================================
    if dataset_name == 'adoption':
        try:
            train_df = pd.read_csv(cfg.data_train_eval_tabular)
            val_df = pd.read_csv(cfg.data_val_eval_tabular)
            test_df = pd.read_csv(cfg.data_test_eval_tabular)
            len_train, len_val = len(train_df), len(val_df)
            full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
        except FileNotFoundError as e:
            print(f"错误：找不到文件 {e.filename}。请确保 config 中的路径正确。")
            sys.exit(1)

        TARGET = 'AdoptionSpeed'
        COLS_TO_DROP =  ['RescuerID', 'Description', 'PetID'] 
        features = [col for col in full_df.columns if col not in [TARGET] + COLS_TO_DROP]
        
        # Adoption 特定的特征识别
        categorical_features = [col for col in features if full_df[col].dtype == 'object' or full_df[col].nunique() < 25]
        numerical_features = [col for col in features if col not in categorical_features]
        print(f"Adoption: 找到 {len(numerical_features)} 个数值特征和 {len(categorical_features)} 个分类特征。")

        # 1. 处理分类特征
        for col in categorical_features:
            full_df[col] = full_df[col].astype(str)
            encoder = LabelEncoder()
            full_df[col] = encoder.fit_transform(full_df[col])
            cat_cardinalities.append(len(encoder.classes_))

        # 2. 处理数值特征
        scaler = StandardScaler()
        scaler.fit(full_df.loc[:len_train-1, numerical_features])
        full_df[numerical_features] = scaler.transform(full_df[numerical_features])

        # 3. 拆分
        train_processed_df = full_df.iloc[:len_train]
        val_processed_df = full_df.iloc[len_train:len_train + len_val]
        test_processed_df = full_df.iloc[len_train + len_val:]

        # 4. 转换为张量
        target_dtype = torch.int64 if cfg.task == 'classification' else torch.float32
        def to_tensors(df):
            return (
                torch.tensor(df[numerical_features].values.astype(np.float32)),
                torch.tensor(df[categorical_features].values.astype(np.int64)),
                torch.tensor(df[TARGET].values.astype(np.int64 if cfg.task == 'classification' else np.float32)) # 动态类型
            )
        
        X_train_num, X_train_cat, y_train = to_tensors(train_processed_df)
        X_val_num, X_val_cat, y_val = to_tensors(val_processed_df)
        X_test_num, X_test_cat, y_test = to_tensors(test_processed_df)
        
        print("Adoption 数据预处理完成。")

    # ==================================================================
    # == dvm 数据集逻辑 (请您填写)
    # ==================================================================
    elif dataset_name == 'dvm':
        print(f"正在加载 {dataset_name} 数据...")
        try:
            # --- 1. 加载 dvm-car 数据 (X 和 y 分开加载) ---
            train_df = pd.read_csv(cfg.data_train_eval_tabular, header=None)
            val_df = pd.read_csv(cfg.data_val_eval_tabular, header=None)
            test_df = pd.read_csv(cfg.data_test_eval_tabular, header=None)
            
            # 修正：使用 torch.load 加载标签
            y_train_tensor = torch.load(cfg.labels_train_eval_tabular, weights_only=False)
            y_val_tensor = torch.load(cfg.labels_val_eval_tabular, weights_only=False)
            y_test_tensor = torch.load(cfg.labels_test_eval_tabular, weights_only=False)

            len_train, len_val = len(train_df), len(val_df)
            
            # 合并特征 (X)
            full_df_X = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
            
            # 合并标签 (y)
            # (假设 .pt 文件中的 tensor 可以用 .numpy() 转换)
            y_train_series = pd.Series(y_train_tensor.numpy() if hasattr(y_train_tensor, 'numpy') else y_train_tensor)
            y_val_series = pd.Series(y_val_tensor.numpy() if hasattr(y_val_tensor, 'numpy') else y_val_tensor)
            y_test_series = pd.Series(y_test_tensor.numpy() if hasattr(y_test_tensor, 'numpy') else y_test_tensor)
            
            full_df_y = pd.concat([y_train_series, y_val_series, y_test_series], axis=0, ignore_index=True)
            
            TARGET = 'dvm' # 给标签列一个名字
            full_df_y.name = TARGET
            
            # 最终合并 X 和 y 以便统一处理
            full_df = pd.concat([full_df_X, full_df_y], axis=1)

        except FileNotFoundError as e:
            print(f"错误：找不到 dvm-car 所需的文件 {e.filename}。请检查 config 中的路径。")
            sys.exit(1)
        except Exception as e:
            print(f"加载 dvm-car 数据时出错: {e}")
            sys.exit(1)

        # 2. dvm-car 特征定义 (根据您的 13 num + 4 cat 新信息)
        print("dvm-car: 正在使用您提供的 13-num, 4-cat 结构...")
        
        # 验证总列数是否符合 13+4=17
        total_features = cfg.get('num_con', 13) + cfg.get('num_cat', 4)
        if full_df_X.shape[1] != total_features:
            print(f"警告：dvm-car 的特征文件有 {full_df_X.shape[1]} 列，但预期是 {total_features} (13+4) 列。")
            # 仍然按 13/4 拆分，但这可能是一个错误
        
        # 从 config 或硬编码获取列数
        num_con_count = cfg.get('num_con', 13)
        num_cat_count = cfg.get('num_cat', 4)

        numerical_features = list(full_df_X.columns[:num_con_count]) # 前 13 列
        categorical_features = list(full_df_X.columns[num_con_count : num_con_count + num_cat_count]) # 后 4 列
        
        print(f"dvm-car: 成功定义 {len(numerical_features)} 个数值特征和 {len(categorical_features)} 个分类特征。")

        if len(categorical_features) != num_cat_count:
             print(f"错误：未能正确提取 {num_cat_count} 个分类特征。请检查数据列数。")
             sys.exit(1)
        
        # 3. dvm-car 预处理 (无预处理，仅计算基数)
        print("dvm-car: 跳过 StandardScaler/LabelEncoder (假设数据已预处理)。")
        cat_cardinalities = []
        if categorical_features:
            print("dvm-car: 正在计算分类特征的基数 (假设为 0-based 编码)...")
            for col in categorical_features:
                # 假设是 0-based 编码，最大值+1 = 基数
                cardinality = int(full_df[col].max()) + 1
                cat_cardinalities.append(cardinality)
                # 确保数据是整数类型
                full_df[col] = full_df[col].astype(np.int64) 
            print(f"dvm-car: 计算得到的基数: {cat_cardinalities}")

        # 4. 拆分并转换为张量
        print("正在拆分数据并转换为 Tensors...")
        train_processed_df = full_df.iloc[:len_train]
        val_processed_df = full_df.iloc[len_train:len_train + len_val]
        test_processed_df = full_df.iloc[len_train + len_val:]

        def to_tensors_dvm(df):
            # 分别从已识别的列表中提取
            num_tensor = torch.tensor(df[numerical_features].values.astype(np.float32))
            cat_tensor = torch.tensor(df[categorical_features].values.astype(np.int64))
            
            return (
                num_tensor,
                cat_tensor,
                torch.tensor(df[TARGET].values.astype(np.int64 if cfg.task == 'classification' else np.float32))
            )

        X_train_num, X_train_cat, y_train = to_tensors_dvm(train_processed_df)
        X_val_num, X_val_cat, y_val = to_tensors_dvm(val_processed_df)
        X_test_num, X_test_cat, y_test = to_tensors_dvm(test_processed_df)

        print("dvm-car 数据预处理完成。")
        
    else:
        print(f"错误: 未知的数据集名称 '{dataset_name}'。请检查 config 文件。")
        sys.exit(1)

    # ==================================================================
    # == 创建 DataLoaders (通用)
    # ==================================================================
    print("--- 2. 正在创建 DataLoaders ---")
    train_dataset = TensorDataset(X_train_num, X_train_cat, y_train)
    val_dataset = TensorDataset(X_val_num, X_val_cat, y_val)
    test_dataset = TensorDataset(X_test_num, X_test_cat, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("DataLoaders 创建完成。")

    # 模型输出维度
    d_out = cfg.num_classes
    
    # 返回所有必要组件
    model_inputs = {
        "n_num_features": len(numerical_features),
        "cat_cardinalities": cat_cardinalities,
        "d_out": d_out,
        "task": cfg.task
    }
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    return loaders, model_inputs

# --- 3. 定义模型创建函数 (已修正：补全 rtdl.Transformer 所需的参数) ---
def create_model(params, n_num_features, cat_cardinalities, d_out, device):
    """
    根据参数创建 FTTransformer 模型实例。
    未在 params 中指定的参数将使用硬编码的默认值。
    """
    base_ft_transformer = rtdl.FTTransformer(
        feature_tokenizer=rtdl.modules.FeatureTokenizer(
            n_num_features=n_num_features, 
            cat_cardinalities=cat_cardinalities, 
            d_token=params['d_token']
        ),
        transformer=rtdl.modules.Transformer(
            # --- 1. 从 params 字典 (搜索空间) 获取 ---
            d_token=params['d_token'],
            n_blocks=params['n_blocks'], 
            attention_dropout=params['attention_dropout'], 
            ffn_d_hidden=params['ffn_d_hidden'], # "FFN Factor"
            ffn_dropout=params['ffn_dropout'], 
            residual_dropout=params['residual_dropout'],
            
            # --- 2. [!] 补全缺失的14个硬编码“默认”参数 ---
            #    (这些是 rtdl 库所必需的)
            attention_n_heads=8,
            attention_initialization='kaiming',
            attention_normalization='LayerNorm',
            ffn_activation='ReLU',
            ffn_normalization='LayerNorm',
            prenormalization=True,
            first_prenormalization=False,
            last_layer_query_idx=[-1],
            n_tokens=None,
            kv_compression_ratio=None,
            kv_compression_sharing=None,
            head_activation=nn.Identity,
            head_normalization=nn.Identity,
            d_out=params['d_token'] # Transformer 内部 d_out == d_token
        ),
    )
    model = MyFTTransformer(
        ft_transformer_module=base_ft_transformer, 
        d_token=params['d_token'], 
        d_out=d_out # 这是最终分类/回归头的 d_out (例如 286)
    ).to(device)
    return model

# --- 4. 辅助函数：获取损失函数和评估指标 ---
def create_loss_fn(task, device):
    if task == 'classification':
        return nn.CrossEntropyLoss().to(device)
    elif task == 'regression':
        return nn.MSELoss().to(device)
    else:
        raise ValueError(f"未知的任务类型: {task}")

def get_scoring_info(task):
    """获取阶段一搜索所需的评估指标和优化方向"""
    if task == 'classification':
        return 'accuracy', 'max' # 指标名称, 优化方向
    elif task == 'regression':
        return 'rmse', 'min'
    else:
        raise ValueError(f"未知的任务类型: {task}")

# --- 5. 定义训练与评估函数 ---

# 阶段一函数：快速搜索
def search_for_best_params(param_combinations, cfg, seed, loaders, model_inputs, device):
    print("\n" + "-"*10 + f" [种子 {seed}] 阶段一：快速超参数搜索 (15 epochs) " + "-"*10)
    
    train_loader, val_loader = loaders['train'], loaders['val']
    n_num, cats, d_out, task = model_inputs.values()
    
    scoring_metric, mode = get_scoring_info(task)
    best_score = -float('inf') if mode == 'max' else float('inf')
    best_params = None
    
    for i, params in enumerate(param_combinations):
        print(f"\n--- [试验 {i+1}/{len(param_combinations)}] ---")
        print(f"测试参数: {params}")
        
        set_seed(seed)
        model = create_model(params, n_num, cats, d_out, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        loss_fn = create_loss_fn(task, device)
        
        # 训练固定的15个epoch
        for epoch in range(15):
            model.train()
            for x_num_batch, x_cat_batch, y_batch in train_loader:
                x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(x_num_batch, x_cat_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        # 在验证集上评估
        model.eval()
        val_preds_proba = []
        val_labels = []
        with torch.no_grad():
            for x_num_batch, x_cat_batch, y_batch in val_loader:
                x_num_batch, x_cat_batch = x_num_batch.to(device), x_cat_batch.to(device)
                y_pred = model(x_num_batch, x_cat_batch)
                
                # 统一处理: proba 用于分类, value 用于回归
                if task == 'classification':
                    val_preds_proba.append(y_pred.softmax(dim=1).cpu().numpy())
                else:
                    val_preds_proba.append(y_pred.cpu().numpy()) # (N,) or (N, 1)
                val_labels.append(y_batch.cpu().numpy())
        
        val_preds_proba = np.concatenate(val_preds_proba)
        val_labels = np.concatenate(val_labels)
        
        # 动态计算得分
        current_score = 0.0
        if task == 'classification':
            val_preds_class = np.argmax(val_preds_proba, axis=1)
            current_score = accuracy_score(val_labels, val_preds_class)
        elif task == 'regression':
            current_score = np.sqrt(mean_squared_error(val_labels, val_preds_proba.squeeze()))
        
        print(f"试验 {i+1} 验证集 {scoring_metric}: {current_score:.4f}")
        
        if (mode == 'max' and current_score > best_score) or \
           (mode == 'min' and current_score < best_score):
            best_score = current_score
            best_params = params
            print(f"  (发现新的最佳参数!)")
            
    print("\n" + "-"*10 + " 阶段一搜索完成 " + "-"*10)
    print(f"最佳验证集 {scoring_metric}: {best_score:.4f}")
    print(f"选定的最佳参数: {best_params}")
    return best_params

# 阶段二函数：使用早停充分训练
def train_final_model(best_params, cfg, seed, loaders, model_inputs, device, 
                      patience: int, max_epochs: int):
    print("\n" + "-"*10 + f" [种子 {seed}] 阶段二：使用早停机制充分训练最佳模型 " + "-"*10)
    
    train_loader, val_loader = loaders['train'], loaders['val']
    n_num, cats, d_out, task = model_inputs.values()
    
    set_seed(seed)
    model = create_model(best_params, n_num, cats, d_out, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    loss_fn = create_loss_fn(task, device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f'best_model_seed_{seed}.pt' 
    # [!] 使用来自 json 的参数
    # max_epochs = cfg.hyperparams.max_epochs (移除)
    # patience = cfg.hyperparams.patience (移除)

    for epoch in range(max_epochs):
        model.train()
        for x_num_batch, x_cat_batch, y_batch in train_loader:
            x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_num_batch, x_cat_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_num_batch, x_cat_batch, y_batch in val_loader:
                x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                y_pred = model(x_num_batch, x_cat_batch)
                val_loss += loss_fn(y_pred, y_batch).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  (验证集损失连续 {patience} 个epoch未改善，触发早停!)")
                break
    
    print(f"加载在验证集上性能最佳的模型 (来自 {best_model_path})...")
    model.load_state_dict(torch.load(best_model_path))
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        
    return model

# 阶段三函数：在测试集上评估
def evaluate_final_model(final_model, test_loader, task, device):
    final_model.eval()
    all_preds_proba, all_labels = [], []
    with torch.no_grad():
        for x_num_batch, x_cat_batch, y_batch in test_loader:
            x_num_batch, x_cat_batch = x_num_batch.to(device), x_cat_batch.to(device)
            
            y_pred = final_model(x_num_batch, x_cat_batch)
            
            if task == 'classification':
                all_preds_proba.append(y_pred.softmax(dim=1).cpu().numpy())
            else:
                all_preds_proba.append(y_pred.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            
    all_preds_proba = np.concatenate(all_preds_proba)
    all_labels = np.concatenate(all_labels)

    metrics_dict = {}
    result_line = ""

    if task == 'classification':
        all_preds_class = np.argmax(all_preds_proba, axis=1)
        acc = accuracy_score(all_labels, all_preds_class)
        auc = roc_auc_score(all_labels, all_preds_proba, multi_class='ovr', average='macro')
        macro_f1 = f1_score(all_labels, all_preds_class, average='macro')
        
        metrics_dict = {'acc': acc, 'auc': auc, 'macro-F1': macro_f1}
        result_line = f"acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}"
        
    elif task == 'regression':
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds_proba.squeeze()))
        metrics_dict = {'rmse': rmse}
        result_line = f"rmse:{rmse:.4f}"

    print(f"最终测试集性能: {result_line}")
    return metrics_dict, result_line

# --- 6. 主执行流程 (Hydra Main) ---
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    
    print("--- 最终配置: ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------")
    print(f"Hydra 工作目录: {os.getcwd()}")
    print("--------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用设备: {device}')
    
    original_cwd = get_original_cwd() 
    model_config_filename = 'ftt_model_config.json' # [!] 文件名修改
    model_config_path = os.path.join(original_cwd, model_config_filename)
    
    print(f"--- 正在从 {model_config_path} 加载模型配置 ---")
    try:
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        search_space = model_config['search_space']
        hyperparams = model_config['hyperparams']
        print("模型配置 (hyperparams 和 search_space) 加载成功。")
    except FileNotFoundError:
        print(f"错误: 找不到模型配置文件: {model_config_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"错误: 模型配置文件 '{model_config_path}' 缺少键: {e}")
        sys.exit(1)
    
    
    # [!] 2. 将 hyperparams 提取到变量
    seeds = list(hyperparams['seeds'])
    batch_size = hyperparams['batch_size']
    D_TOKEN = hyperparams['d_token']
    LEARNING_RATE = hyperparams['learning_rate']
    WEIGHT_DECAY = hyperparams['weight_decay']
    N_TRIALS = hyperparams['n_trials']
    patience = hyperparams['patience']
    max_epochs = hyperparams['max_epochs']

    final_results_summary = []

    # [!] 3. 加载数据, 传入 batch_size
    loaders, model_inputs = load_and_preprocess_data(cfg, batch_size)

    for seed in seeds:
        print("\n" + "="*30 + f" 开始执行，随机种子: {seed} " + "="*30)
        set_seed(seed)

        # 4. 生成随机超参数组合 (使用来自 JSON 的 search_space)
        param_combinations = []
        for _ in range(N_TRIALS):
            params = {
                'n_blocks': random.choice(search_space['n_blocks']),
                'ffn_d_hidden': random.choice(search_space['ffn_d_hidden']),
                'residual_dropout': random.uniform(*search_space['residual_dropout']),
                'attention_dropout': random.uniform(*search_space['attention_dropout']),
                'ffn_dropout': random.uniform(*search_space['ffn_dropout']),
                'd_token': D_TOKEN, 
                'learning_rate': LEARNING_RATE, 
                'weight_decay': WEIGHT_DECAY,
            }
            param_combinations.append(params)
        
        # 5. 阶段一：搜索
        best_params = search_for_best_params(
            param_combinations, cfg, seed, loaders, model_inputs, device
        )
        
        # [!] 6. 阶段二：训练, 传入 patience 和 max_epochs
        final_model = train_final_model(
            best_params, cfg, seed, loaders, model_inputs, device,
            patience=patience, max_epochs=max_epochs
        )
        
        # 7. 阶段三：评估
        print("\n" + "-"*10 + f" [种子 {seed}] 阶段三：在测试集上进行最终评估 " + "-"*10)
        metrics_dict, result_line = evaluate_final_model(
            final_model, loaders['test'], model_inputs['task'], device
        )
        
        result_dict = {
            'seed': seed, 
            'best_params': best_params, 
            'result_line': result_line,
            **metrics_dict
        }
        final_results_summary.append(result_dict)

    # --- 7. 最终总结 ---
    print("\n\n" + "="*30 + " 所有实验最终总结 " + "="*30)
    
    output_file_path = to_absolute_path(cfg.output_filename)
    print(f"准备将结果写入到: {output_file_path}")

    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, 'w') as f:
        f.write(f"--- 实验配置 (来自 config.yaml): {cfg.target} ---\n")
        f.write(OmegaConf.to_yaml(cfg)) # 写入数据配置
        f.write("\n--- 模型配置 (来自 ftt_model_config.json) ---\n")
        f.write(json.dumps(model_config, indent=2)) # [!] 写入模型配置
        f.write("\n\n" + "="*30 + " 所有实验最终总结 " + "="*30 + "\n")
        
        for final_result in final_results_summary:
            result_line = f"种子: {final_result['seed']} | {final_result['result_line']}"
            print(result_line)
            f.write(result_line + "\n")

        params_header = f"\n最佳参数的例子 (来自最后一个种子 {final_results_summary[-1]['seed']}):"
        params_details = str(final_results_summary[-1]['best_params'])
        
        print(params_header)
        print(params_details)
        f.write(params_header + "\n")
        f.write(params_details + "\n")
        f.write("="*80 + "\n")

    print(f"\n结果已成功写入到文件: {output_file_path}")


if __name__ == "__main__":
    main()