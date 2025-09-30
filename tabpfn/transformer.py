import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import rtdl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import time
import os
import random

# --- 0. 配置与设置随机种子 ---
# 自动选择设备 (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'正在使用设备: {device}')

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 数据路径 ---
train_path = '/data1/jiazy/tab_image_bench/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_train.csv'
val_path = '/data1/jiazy/tab_image_bench/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_valid.csv'
test_path = '/data1/jiazy/tab_image_bench/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_test.csv'
output_file_path = '/data0/jiazy/tibench/result/fttransformer.txt'

# --- 自定义模型类 ---
class MyFTTransformer(nn.Module):
    def __init__(self, ft_transformer_module, d_token, d_out):
        super().__init__()
        self.ft_transformer = ft_transformer_module
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num, x_cat):
        x = self.ft_transformer(x_num, x_cat)
        x = self.head(x)
        return x

# --- 1. 数据加载与预处理 ---
# --- 1. 数据加载与预处理 (已完全修正) ---
print("开始加载和预处理数据...")
try:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    # 记录原始长度以便之后拆分
    len_train, len_val = len(train_df), len(val_df)
    
    # 合并所有数据集以统一处理
    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。请确保文件路径正确。")
    exit()

TARGET = 'AdoptionSpeed'
COLS_TO_DROP = ['Name', 'RescuerID', 'Description', 'PetID']
features = [col for col in full_df.columns if col not in [TARGET] + COLS_TO_DROP]
categorical_features = [col for col in features if full_df[col].dtype == 'object' or full_df[col].nunique() < 25]
numerical_features = [col for col in features if col not in categorical_features]

# --- 正确的处理流程 ---
# 1. 处理分类特征 (在所有数据上fit_transform)
cat_cardinalities = []
for col in categorical_features:
    full_df[col] = full_df[col].astype(str)
    encoder = LabelEncoder()
    full_df[col] = encoder.fit_transform(full_df[col])
    cat_cardinalities.append(len(encoder.classes_))

# 2. 处理数值特征 (仅在训练数据上fit，然后transform所有数据)
scaler = StandardScaler()
# 从合并后的数据中定位训练集部分进行fit
scaler.fit(full_df.loc[:len_train-1, numerical_features])
# 对整个数据集进行transform
full_df[numerical_features] = scaler.transform(full_df[numerical_features])

# 3. 将处理好的 full_df 拆分回训练、验证和测试集
train_processed_df = full_df.iloc[:len_train]
val_processed_df = full_df.iloc[len_train:len_train + len_val]
test_processed_df = full_df.iloc[len_train + len_val:]

# 4. 定义一个函数，从这些处理好的DataFrame创建张量
def to_tensors(df):
    return (
        torch.tensor(df[numerical_features].values.astype(np.float32)),
        torch.tensor(df[categorical_features].values.astype(np.int64)),
        torch.tensor(df[TARGET].values.astype(np.int64))
    )

# 5. 使用处理好的DataFrame创建最终的张量
X_train_num, X_train_cat, y_train = to_tensors(train_processed_df)
X_val_num, X_val_cat, y_val = to_tensors(val_processed_df)
X_test_num, X_test_cat, y_test = to_tensors(test_processed_df)

print("数据预处理完成。")

# --- 2. 创建 PyTorch DataLoaders ---
batch_size = 256
train_dataset = TensorDataset(X_train_num, X_train_cat, y_train)
val_dataset = TensorDataset(X_val_num, X_val_cat, y_val)
test_dataset = TensorDataset(X_test_num, X_test_cat, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 3. 定义模型创建函数 ---
def create_model(params):
    base_ft_transformer = rtdl.FTTransformer(
        feature_tokenizer=rtdl.modules.FeatureTokenizer(n_num_features=len(numerical_features), cat_cardinalities=cat_cardinalities, d_token=params['d_token']),
        transformer=rtdl.modules.Transformer(d_token=params['d_token'], n_blocks=params['n_blocks'], attention_n_heads=8, attention_dropout=params['attention_dropout'], ffn_d_hidden=params['ffn_d_hidden'], ffn_dropout=params['ffn_dropout'], residual_dropout=params['residual_dropout'], attention_initialization='kaiming', attention_normalization='LayerNorm', ffn_activation='ReLU', ffn_normalization='LayerNorm', prenormalization=True, first_prenormalization=False, last_layer_query_idx=[-1], n_tokens=None, kv_compression_ratio=None, kv_compression_sharing=None, head_activation=nn.Identity, head_normalization=nn.Identity, d_out=params['d_token']),
    )
    model = MyFTTransformer(ft_transformer_module=base_ft_transformer, d_token=params['d_token'], d_out=5).to(device)
    return model

# --- 4. 定义训练与评估函数 ---

# 阶段一函数：快速搜索
def search_for_best_params(param_combinations, seed):
    print("\n" + "-"*10 + " 阶段一：快速超参数搜索 (15 epochs) " + "-"*10)
    best_val_acc = -1
    best_params = None
    
    for i, params in enumerate(param_combinations):
        print(f"\n--- [种子 {seed}] - [试验 {i+1}/{len(param_combinations)}] ---")
        print(f"测试参数: {params}")
        
        set_seed(seed)
        model = create_model(params)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        loss_fn = nn.CrossEntropyLoss()
        
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
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for x_num_batch, x_cat_batch, y_batch in val_loader:
                x_num_batch, x_cat_batch = x_num_batch.to(device), x_cat_batch.to(device)
                y_pred = model(x_num_batch, x_cat_batch)
                val_preds.append(y_pred.cpu().numpy())
                val_labels.append(y_batch.cpu().numpy())
        
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_acc = accuracy_score(val_labels, np.argmax(val_preds, axis=1))
        print(f"试验 {i+1} 验证集准确率: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            print(f"  (发现新的最佳参数!)")
            
    print("\n" + "-"*10 + " 阶段一搜索完成 " + "-"*10)
    print(f"最佳验证集准确率: {best_val_acc:.4f}")
    print(f"选定的最佳参数: {best_params}")
    return best_params

# 阶段二函数：使用早停充分训练
def train_final_model(best_params, seed):
    print("\n" + "-"*10 + " 阶段二：使用早停机制充分训练最佳模型 " + "-"*10)
    set_seed(seed)
    model = create_model(best_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_path = f'best_model_seed_{seed}.pt'
    max_epochs = 100 # 提高上限，让早停决定

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
                print(f"  (验证集损失连续 {patience} 个epoch未改善，触发早停!)")
                break
    
    print("加载在验证集上性能最佳的模型...")
    model.load_state_dict(torch.load(best_model_path))
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        
    return model

# --- 5. 主执行流程 ---
search_space = {
    'n_blocks': [1, 2, 3, 4], 'ffn_d_hidden': [64, 128, 256, 512],
    'residual_dropout': (0.0, 0.2), 'attention_dropout': (0.0, 0.5), 'ffn_dropout': (0.0, 0.5),
}
N_TRIALS = 10
D_TOKEN = 192
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

seeds = [2024, 2025, 2026]
final_results_summary = []

for seed in seeds:
    print("\n" + "="*30 + f" 开始执行，随机种子: {seed} " + "="*30)
    set_seed(seed)

    # 生成随机超参数组合
    param_combinations = []
    for _ in range(N_TRIALS):
        params = {
            'n_blocks': random.choice(search_space['n_blocks']),
            'ffn_d_hidden': random.choice(search_space['ffn_d_hidden']),
            'residual_dropout': random.uniform(*search_space['residual_dropout']),
            'attention_dropout': random.uniform(*search_space['attention_dropout']),
            'ffn_dropout': random.uniform(*search_space['ffn_dropout']),
            'd_token': D_TOKEN, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        }
        param_combinations.append(params)
    
    # 阶段一：搜索最佳参数
    best_params = search_for_best_params(param_combinations, seed)
    
    # 阶段二：用最佳参数充分训练模型
    final_model = train_final_model(best_params, seed)
    
    # 阶段三：在测试集上评估最终模型
    print("\n" + "-"*10 + f" [种子 {seed}] 阶段三：在测试集上进行最终评估 " + "-"*10)
    final_model.eval()
    all_preds_proba, all_labels = [], []
    with torch.no_grad():
        for x_num_batch, x_cat_batch, y_batch in test_loader:
            x_num_batch, x_cat_batch = x_num_batch.to(device), x_cat_batch.to(device)
            y_pred_proba = final_model(x_num_batch, x_cat_batch).softmax(dim=1)
            all_preds_proba.append(y_pred_proba.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            
    all_preds_proba = np.concatenate(all_preds_proba)
    all_labels = np.concatenate(all_labels)
    all_preds_class = np.argmax(all_preds_proba, axis=1)

    acc = accuracy_score(all_labels, all_preds_class)
    auc = roc_auc_score(all_labels, all_preds_proba, multi_class='ovr', average='macro')
    macro_f1 = f1_score(all_labels, all_preds_class, average='macro')
    
    result_dict = {'seed': seed, 'acc': acc, 'auc': auc, 'macro-F1': macro_f1, 'best_params': best_params}
    final_results_summary.append(result_dict)
    
    print(f"最终测试集性能: acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}")

# --- 6. 最终总结 ---
print("\n\n" + "="*30 + " 所有实验最终总结 " + "="*30)
for final_result in final_results_summary:
    print(f"种子: {final_result['seed']} | "
          f"Acc: {final_result['acc']:.4f} | "
          f"AUC: {final_result['auc']:.4f} | "
          f"Macro-F1: {final_result['macro-F1']:.4f}")
print(f"最佳参数的例子 (来自最后一个种子): {final_results_summary[-1]['best_params']}")
print("="*80)

# 确保结果目录存在
output_dir = os.path.dirname(output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将总结写入文件
with open(output_file_path, 'w') as f:
    f.write("="*30 + " 所有实验最终总结 " + "="*30 + "\n")
    for final_result in final_results_summary:
        # 准备用于打印和写入的字符串
        result_line = (f"种子: {final_result['seed']} | "
                       f"Acc: {final_result['acc']:.4f} | "
                       f"AUC: {final_result['auc']:.4f} | "
                       f"Macro-F1: {final_result['macro-F1']:.4f}")
        
        print(result_line)
        f.write(result_line + "\n")

    # 准备参数详情用于打印和写入
    params_header = f"\n最佳参数的例子 (来自最后一个种子 {final_results_summary[-1]['seed']}):"
    params_details = str(final_results_summary[-1]['best_params'])
    
    print(params_header)
    print(params_details)
    f.write(params_header + "\n")
    f.write(params_details + "\n")

    end_line = "="*80
    print(end_line)
    f.write(end_line + "\n")

print(f"\n结果已成功写入到文件: {output_file_path}")
