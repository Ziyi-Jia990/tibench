import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

class PetFinderDataset(TensorDataset):
    """
    一個簡單的包裝，用於確保回傳的是 (image, label) 元組。
    """
    def __init__(self, images_tensor, labels_tensor):
        super().__init__(images_tensor, labels_tensor)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

class PetFinderDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning 的資料模組，用於組織資料載入。
    """
    def __init__(self, data_dir, batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 0

    def setup(self, stage=None):
        """載入資料並創建 Dataset。"""
        # 載入預處理好的 Tensor
        # ⚠️ 注意：這裡仍然是載入 'images_train.pt'，用於 ResNet 訓練
        print(f"--- 正在從 {self.data_dir} 載入圖片資料 ---")
        train_images = torch.load(os.path.join(self.data_dir, 'images_train.pt'))
        train_labels = torch.load(os.path.join(self.data_dir, 'labels_train.pt'))
        test_images = torch.load(os.path.join(self.data_dir, 'images_test.pt'))
        test_labels = torch.load(os.path.join(self.data_dir, 'labels_test.pt'))

        # 創建 TensorDataset
        self.train_dataset = PetFinderDataset(train_images, train_labels)
        self.test_dataset = PetFinderDataset(test_images, test_labels)

        # 自動計算類別數量
        self.num_classes = len(torch.unique(train_labels))
        print(f"資料集準備完畢。訓練集大小: {len(self.train_dataset)}, 測試集大小: {len(self.test_dataset)}")
        print(f"檢測到 {self.num_classes} 個類別。")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

class SupervisedResNet(pl.LightningModule):
    """
    PyTorch Lightning 的模型模組，用於監督學習。
    """
    def __init__(self, num_classes, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # 使用新的 API
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        # 確保指標計算在正確的設備上
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        self.test_acc.update(logits, labels)
        self.test_auc.update(probs, labels)
        self.test_f1.update(logits, labels)
        self.log_dict({'test_loss': loss, 'test_acc': self.test_acc, 'test_auc': self.test_auc, 'test_macro_f1': self.test_f1}, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

if __name__ == '__main__':
    
    
    # --- 1. 設定命令列參數 ---
    parser = argparse.ArgumentParser(description='Train a supervised ResNet on PetFinder dataset.')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--bs', '--batch_size', type=int, default=64, help='Actual batch size that fits into memory.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--data_dir', type=str, default='/home/debian/jzy/dataset/petfinder_adoptionprediction', help='Directory containing the .pt dataset files.')
    # --- 新增梯度累積參數 ---
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Number of steps for gradient accumulation to simulate a larger batch size.')
    parser.add_argument('--seed', type=int, default=2026, help='random seed')

    args = parser.parse_args()
    print(f"命令行參數: {args}")
    pl.seed_everything(args.seed) # 為了可複線性

    # --- 2. 初始化資料模組 ---
    datamodule = PetFinderDataModule(data_dir=args.data_dir, batch_size=args.bs)
    datamodule.setup()

    # --- 3. 初始化模型 ---
    model = SupervisedResNet(num_classes=datamodule.num_classes, learning_rate=args.lr)

    # --- 4. 設定模型保存回調 ---
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='resnet-supervised-{epoch:02d}-{train_loss:.2f}',
        monitor='train_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    # --- 5. 初始化並執行訓練器 ---
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=True,
        # --- 使用梯度累積參數 ---
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    effective_bs = args.bs * args.accumulate_grad_batches
    print(f"\n================== 訓練設定 ==================")
    print(f"實際批次大小 (Batch Size): {args.bs}")
    print(f"梯度累積步數 (Accumulation Steps): {args.accumulate_grad_batches}")
    print(f"==> 有效批次大小 (Effective Batch Size): {effective_bs}")
    print("==============================================\n")

    trainer.fit(model, datamodule=datamodule)
    
    print("\n================== 測試最佳模型 ==================")
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path='best')
    print("================== 測試結果 ==================")
    print(test_results)