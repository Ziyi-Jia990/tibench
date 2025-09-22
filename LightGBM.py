import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
# 导入新增的评估指标
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

# --- 1. 数据准备 ---
# 直接从您提供的文件路径加载数据
train_path = '/home/admin_czj/jiazy/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_train.csv'
test_path = '/home/admin_czj/jiazy/PetFinder_datasets/dataset/petfinder_adoptionprediction/dataset_test.csv'

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
COLS_TO_DROP = ['Name', 'RescuerID', 'Description', 'PetID']

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


# --- 2. 定义 LightGBM 模型和参数网格 ---
lgbm = lgb.LGBMClassifier(
    objective='multiclass', 
    num_class=5, 
    random_state=42
)

param_grid = {
    'num_leaves': [31, 127],
    'learning_rate': [0.01, 0.1],
    'min_child_samples': [20, 50, 100],
    'min_sum_hessian_in_leaf': [1e-3, 1e-2, 1e-1]
}


# --- 3. 使用 GridSearchCV 进行参数调优 ---
print("开始进行网格搜索以寻找最佳超参数...")
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
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


# --- 5. 在测试集上评估最佳模型 (已按要求修改输出) ---
print("使用最佳模型在测试集上进行最终评估...")
best_model = grid_search.best_estimator_

# 进行预测
y_pred = best_model.predict(X_test)
# AUC 计算需要概率输出
y_pred_proba = best_model.predict_proba(X_test)

# 计算所需指标
# 1. 准确率 (Accuracy)
acc = accuracy_score(y_test, y_pred)

# 2. 宏平均 F1-score (Macro-F1)
macro_f1 = f1_score(y_test, y_pred, average='macro')

# 3. 宏平均 One-vs-Rest AUC
#    对于多分类问题，需要提供概率且指定 multi_class='ovr'
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

# 按照您指定的格式输出
print("评估结果:")
print(f"acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}")