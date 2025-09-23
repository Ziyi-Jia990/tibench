import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

# --- 1. 数据准备 ---
# 直接从您提供的文件路径加载数据
train_path = '/root/autodl-tmp/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_train.csv'
test_path = '/root/autodl-tmp/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_test.csv'

print(f"正在从以下路径加载训练数据: {train_path}")
try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：找不到文件。请确保文件路径正确且文件存在。")
    exit()

# 定义目标变量和需要剔除的列
TARGET = 'AdoptionSpeed'
COLS_TO_DROP = ['RescuerID', 'Description', 'PetID']

# 准备训练数据
y_train = train_df[TARGET]
X_train = train_df.drop(columns=[TARGET] + COLS_TO_DROP)

# 准备测试数据
y_test = test_df[TARGET]
X_test = test_df.drop(columns=[TARGET] + COLS_TO_DROP)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print("-" * 30)

# --- 特征处理：识别并转换分类特征 ---
categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
known_categorical = ['Type', 'Gender', 'Color1', 'Color2', 'Color3', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'Breed1', 'Breed2']
for col in known_categorical:
    if col in X_train.columns and col not in categorical_features:
        categorical_features.append(col)
        
print(f"识别出的分类特征: {categorical_features}")

for col in categorical_features:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

print("分类特征转换完成。")
print("-" * 30)

# --- 新增：定义要遍历的随机种子和输出文件名 ---
random_seeds = [2023, 2025, 2026]
output_file = 'lgb.txt'

# --- 新增：打开文件准备写入结果 ---
with open(output_file, 'w') as f:
    # --- 新增：开始循环遍历所有随机种子 ---
    for seed in random_seeds:
        print(f"\n{'='*25} 正在运行，随机种子: {seed} {'='*25}")
        
        # --- 2. 定义 LightGBM 模型和参数网格 ---
        # 修改：使用循环中的 seed 作为 random_state
        lgbm = lgb.LGBMClassifier(
            objective='multiclass', 
            num_class=5, 
            random_state=seed 
        )

        param_grid = {
            'num_leaves': [31, 127],
            'learning_rate': [0.01, 0.1],
            'min_child_samples': [20, 50, 100],
            'min_sum_hessian_in_leaf': [1e-3, 1e-2, 1e-1]
        }
        
        # --- 3. 使用 GridSearchCV 进行参数调优 ---
        print("开始进行网格搜索以寻找最佳超参数...")
        
        # 修改：为了使交叉验证的划分可复现，使用 StratifiedKFold
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        grid_search = GridSearchCV(
            estimator=lgbm,
            param_grid=param_grid,
            scoring='accuracy',
            cv=cv_splitter,  # 使用新的可复现的cv划分器
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print("网格搜索完成！")
        print("-" * 30)

        # --- 4. 输出最佳参数和分数 ---
        print(f"找到的最佳超参数: {grid_search.best_params_}")
        print(f"在交叉验证中的最佳准确率: {grid_search.best_score_:.4f}")
        print("-" * 30)

        # --- 5. 在测试集上评估最佳模型 ---
        print("使用最佳模型在测试集上进行最终评估...")
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

        # 准备要输出的结果行
        result_line = f"acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}"
        
        # 在控制台打印结果
        print("评估结果:")
        print(result_line)

        # --- 新增：按要求写入到 lgb.txt 文件 ---
        print(f"正在将种子 {seed} 的结果写入到 {output_file}...")
        f.write(f"seed:{seed}\n")
        f.write(result_line + "\n")

print(f"\n所有任务完成！结果已全部保存在文件 '{output_file}' 中。")