import os
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanSquaredError
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


import ot
import rtdl
import random

from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import AUROC, F1Score
from torchmetrics import Accuracy

import sys
sys.path.append("..") 
from datasets.CHARMS_dataset import PetFinderConCatImageDataset


def _debug_finite(tag: str, t: torch.Tensor):
    # 返回 (nan个数, inf个数)，并打印一行摘要
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    print(f"[DEBUG] {tag}: shape={tuple(t.shape)} nan={n_nan} inf={n_inf}")
    return n_nan, n_inf


def check_model_finite(model: nn.Module, tag="backbone"):
    bad = False
    for n, p in model.named_parameters():
        if p is None: 
            continue
        if torch.isnan(p).any() or torch.isinf(p).any():
            print(f"[CHECK][{tag}] PARAM BAD: {n}  nan={torch.isnan(p).sum().item()} inf={torch.isinf(p).sum().item()}")
            bad = True
            break
    if not bad:
        for n, b in model.named_buffers():
            if torch.isnan(b).any() or torch.isinf(b).any():
                print(f"[CHECK][{tag}] BUFFER BAD: {n}  nan={torch.isnan(b).sum().item()} inf={torch.isinf(b).sum().item()}")
                bad = True
                break
    if not bad:
        print(f"[CHECK][{tag}] all params & buffers finite.")
    return not bad


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ImageClassifier(nn.Module):
    def __init__(self, img_reduction_dim: int, model_name: str = 'resnet',
                 out_dims: int = 5,
                 n_num_features: int = 0,
                 cat_cardinalities: list = [],
                 d_token: int = 8, ):
        super().__init__()
        # random.seed(42)
        self.model_name = model_name
        if model_name == "resnet":
            backbone = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
            in_dims = backbone.fc.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.fc = Identity()
        elif model_name == "densenet":
            backbone = models.densenet121(weights=models.densenet.DenseNet121_Weights.IMAGENET1K_V1)
            in_dims = backbone.classifier.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.classifier = Identity()
        elif model_name == "inception":
            backbone = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
            in_dims = backbone.fc.in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.fc = Identity()
        elif model_name == "mobilenet":
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_dims = backbone.classifier[-1].in_features
            img_fc = nn.Sequential(
                nn.Linear(in_dims, 1024),
                nn.Linear(1024, out_dims)
            )
            backbone.classifier[-1] = Identity()

        self.in_dims = in_dims
        self.img_reduction_dim = img_reduction_dim
        self.table_dim = n_num_features + len(cat_cardinalities)

        # con_fc
        linears = []
        for i in range(n_num_features):
            linears.append(nn.Linear(in_dims, 1))
        self.con_fc = nn.ModuleList(linears)
        self.con_fc_num = self.con_fc.__len__()

        # cat_fc
        linears = []
        for i in range(len(cat_cardinalities)):
            linears.append(nn.Linear(in_dims, cat_cardinalities[i]))
        self.cat_fc = nn.ModuleList(linears)
        self.cat_fc_num = self.cat_fc.__len__()

        self.tab_model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            attention_dropout=0.1,
            n_blocks=2,
            ffn_d_hidden=6,
            ffn_dropout=0.2,
            residual_dropout=0.0,
            # last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=out_dims,
        )

        print("self.tab_model: \n", self.tab_model)

        self.backbone = backbone
        self.img_fc = img_fc

        # self.mask = self.get_mask('res_tmp/cluster_res_CelebA_Updating.txt', 'res_tmp/OToutput40_CelebA_Updating.txt')
        self.mask = torch.ones((self.table_dim, in_dims), dtype=torch.long)

    def forward(self, img, tab_con, tab_cat):
        mask = self.mask.to(img.device)
        extracted_feats = self.backbone(img)
        # Try to freeze the backbone ?
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        img_out = self.img_fc(extracted_feats)

        con_out = []
        for i in range(self.con_fc_num):
            masked_feat = mask[i] * extracted_feats
            con_out.append(self.con_fc[i](masked_feat).squeeze(-1))

        cat_out = []
        for i in range(self.cat_fc_num):
            masked_feat = mask[self.con_fc_num + i] * extracted_feats
            cat_out.append(self.cat_fc[i](masked_feat))

        if self.con_fc_num == 0:
            tab_con = None
        if self.cat_fc_num == 0:
            tab_cat = None
        table_features_embed, table_embed_out = self.tab_model(tab_con, tab_cat)
        # print("table_features_embed.shape, table_embed_out.shape:", table_features_embed.shape, table_embed_out.shape)

        return img_out, con_out, cat_out, table_embed_out

    def compute_OT(self, dataset, device):
        test_table_feat, test_channel_feat = self.getTableChannelFeat(dataset, device)
        CostMatrix = self.getCostMatrix(test_table_feat, test_channel_feat)
        P, W = self.compute_coupling(test_table_feat, test_channel_feat, CostMatrix)

        OTOutFileName = 'res_tmp/OToutput_' + str(self.img_reduction_dim) + '_Adoption_Updating.txt'
        np.savetxt(OTOutFileName, P)

        return P

    def get_mask(self, cluster_path: str = None, OT_path: str = None):
        cluster_dict = {}
        with open(cluster_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = list(line[1:-2].split(","))
                cluster_dict[idx] = np.array(line, dtype=int)

        OT = np.loadtxt(OT_path)

        img_dim = self.in_dims
        mask = np.zeros((self.table_dim, img_dim))
        for i in range(self.table_dim):
            for idx_OT, j in enumerate(OT[i]):
                channel_id = cluster_dict[idx_OT]
                if j != 0:
                    mask[i, channel_id] = 1
                else:
                    mask[i, channel_id] = 0
        return torch.tensor(mask, dtype=torch.long)  # np.array (39, 2048)

    def getTableChannelFeat(self, dataset, device):
        resnet = self.backbone
        tab_model = self.tab_model

        test_channel_feat = []
        test_table_feat = []
        index = 0
        for index, row in enumerate(dataset):
            feats, _ = row
            table_features_con = feats[0]
            table_features_cat = feats[1]
            image = feats[2]

            table_features_con = table_features_con.unsqueeze(0).to(device)
            table_features_cat = table_features_cat.unsqueeze(0).to(device)
            image = image.unsqueeze(0).to(device)

            channel_feat = self.getChannelFeature(resnet, image)
            table_feat = self.getTableFeature(tab_model, table_features_con, table_features_cat)

            test_channel_feat.append(channel_feat.unsqueeze(1))
            test_table_feat.append(table_feat.unsqueeze(1))

        print("index: ", index)

        test_channel_feat = torch.cat(test_channel_feat, dim=1)
        test_table_feat = torch.cat(test_table_feat, dim=1)
        return test_table_feat, test_channel_feat

    

    def getChannelFeature(self, resnet, image=None):
        resnet.eval()
        if self.model_name == "mobilenet":
            new_resnet = nn.Sequential(*list(resnet.children())[:-1])
        else:
            new_resnet = nn.Sequential(*list(resnet.children())[:-2])
        # channel_feat = new_resnet(image)  # [1, 2048, 7, 7]
        for i, layer in enumerate(new_resnet):
            image = layer(image)
            if torch.isnan(image).any():
                print(f"NaN detected in layer {i} of the model")
                return None

        channel_feat = image
        channel_feat = channel_feat.squeeze(0)
        channel_feat = channel_feat.reshape((self.in_dims, -1)).detach().cpu().numpy()  # (2048, 7 * 7)

        return torch.tensor(channel_feat, dtype=torch.float)  # (2048, 49)

    def getTableFeature(self, model, table_features_con, table_features_cat):
        model.eval()
        if self.con_fc_num == 0:
            table_features_con = None
        if self.cat_fc_num == 0:
            table_features_cat = None
        table_features_embed, _ = self.tab_model(table_features_con, table_features_cat)
        return table_features_embed.squeeze(0)

    def getCostMatrix(self, test_table_feat, test_channel_feat, out_dir: str = "res_tmp"):
        n_bad = torch.isnan(test_channel_feat).sum().item() + torch.isinf(test_channel_feat).sum().item()
        print(f"[DEBUG] before KMeans: channel_feat nan+inf total = {n_bad}")

        os.makedirs(out_dir, exist_ok=True)  # 新增：确保目录存在
        out_path = os.path.join(out_dir, "cluster_centering_.txt")  # 新增：用可控目录

        # src_x.shape: (table_feat_num, num, table_embed)  tar_x.shape: (img_feat_num, num, img_embed)
        src_x, tar_x = test_table_feat.detach().cpu().numpy(), test_channel_feat.detach().cpu().numpy()
        img_embed = tar_x.shape[2]
        tar_x = tar_x.reshape((self.in_dims, -1))

        print(f"tar_x:Indices of NaNs: {np.where(np.isnan(tar_x)) }")

        kmeans = KMeans(n_clusters=self.img_reduction_dim, random_state=0, n_init="auto").fit(tar_x)
        channel_feat_cluster = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
        tar_x = kmeans.cluster_centers_.reshape((self.img_reduction_dim, -1, img_embed))
        with open(out_path, mode='w') as f:
            for i in range(self.img_reduction_dim):
                f.write(str(tar_x[i]) + '\n')

        labels = kmeans.labels_
        OutFileName = 'res_tmp/cluster_res_' + str(self.img_reduction_dim) + '_Adoption_Updating.txt'
        with open(OutFileName, 'w') as f:
            for i in range(self.img_reduction_dim):
                f.write(str(np.where(labels == i)[0].tolist()) + '\n')

        cost = np.zeros((src_x.shape[0], tar_x.shape[0]))
        for i in range(src_x.shape[0]):
            src_x_similarity_i = src_x[i] / np.linalg.norm(src_x[i])
            src_x_similarity_i = np.dot(src_x_similarity_i, src_x_similarity_i.transpose(1, 0))
            for j in range(tar_x.shape[0]):
                tar_x_similarity_j = tar_x[j] / np.linalg.norm(tar_x[j])
                tar_x_similarity_j = np.dot(tar_x_similarity_j, tar_x_similarity_j.transpose(1, 0))
                # print(src_x_similarity_i.shape, tar_x_similarity_j.shape)
                cost[i, j] = ((src_x_similarity_i - tar_x_similarity_j) ** 2).sum()
                # cost[i, j] = (np.abs(src_x_similarity_i - tar_x_similarity_j)).sum()
        return cost

    def compute_coupling(self, X_src, X_tar, Cost):
        # P = ot.bregman.sinkhorn(ot.unif(X_src.shape[0]), ot.unif(40), Cost, 0.001, numItermax=100000)
        P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(self.img_reduction_dim), Cost, numItermax=100000)
        # P = 0
        W = np.sum(P * np.array(Cost))

        return P, W


class ImageModelPetFinderWithRTDL(pl.LightningModule):
    def __init__(self, hparams, n_num_features, cat_cardinalities, reverse=False, img_reduction_dim=40):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net_img_clf = ImageClassifier(model_name='resnet', n_num_features=n_num_features, cat_cardinalities=cat_cardinalities,
                                           img_reduction_dim=img_reduction_dim)
        self.test_acc = Accuracy(task="multiclass", num_classes=5)
        self.reverse = reverse
        # self.hparams = hparams
        self.img_reduction_dim = img_reduction_dim
        self.valid_loader = self.val_dataloader()
        self.loss_weight_dict = {'con_loss': 0.03, 'cat_loss': 0.03, 'tab_loss': 0.6, 'img_loss': 1}
        # 获取类别数，这是多分类指标所必需的
        num_classes = hparams.num_classes 

        # --- 初始化您的所有测试指标 ---
        if self.hparams.task == 'classification':
            print("Initializing classification metrics: Acc, AUC, Macro-F1")
            num_classes = self.hparams.num_classes
            self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_auc = AUROC(task="multiclass", num_classes=num_classes)
            self.test_macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        
        elif self.hparams.task == 'regression':
            print("Initializing regression metrics: RMSE, MAE, R2Score")
            # 对于RMSE，我们可以先计算MSE，最后再开方
            self.test_mse = MeanSquaredError()
            self.test_mae = MeanAbsoluteError()
            self.test_r2 = R2Score()
            
        else:
            raise ValueError(f"Unsupported task type: {self.hparams.task}. Must be 'classification' or 'regression'.")

    def val_dataloader(self):
        valid_dataset = PetFinderConCatImageDataset(self.hparams.data_val_tabular,
                                                    self.hparams.data_train_eval_imaging)
        valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=8, shuffle=False)
        return valid_loader

    def training_step(self, batch, batch_idx):
        (table_features_con, table_features_cat, image_features), label = batch
        img_out, con_out, cat_out, table_embed_out = self.net_img_clf(image_features, table_features_con,
                                                                      table_features_cat)

        img_loss = F.cross_entropy(img_out, label)

        con_loss = []
        for idx, out_t in enumerate(con_out):
            con_loss.append(F.mse_loss(out_t, table_features_con[:, idx]))
        con_loss_mean = torch.stack(con_loss, dim=0).mean(dim=0) if table_features_con.shape[1] != 0 else 0

        cat_loss = []
        for idx, out_t in enumerate(cat_out):
            cat_loss.append(F.cross_entropy(out_t, table_features_cat[:, idx]))
        cat_loss_mean = torch.stack(cat_loss, dim=0).mean(dim=0) if table_features_cat.shape[1] != 0 else 0

        loss = self.loss_weight_dict['img_loss'] * img_loss + self.loss_weight_dict['con_loss'] * con_loss_mean \
               + self.loss_weight_dict['cat_loss'] * cat_loss_mean

        table_embed_loss = F.cross_entropy(table_embed_out, label)
        loss = loss + self.loss_weight_dict['tab_loss'] * table_embed_loss

        self.log("img_loss", img_loss)
        self.log("tab_con_loss", con_loss_mean)
        self.log("tab_cat_loss", cat_loss_mean)
        self.log("tab_embed_loss", table_embed_loss)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (table_features_con, table_features_cat, image_features), label = batch
        img_out, con_out, cat_out, table_embed_out = self.net_img_clf(image_features, table_features_con,
                                                                      table_features_cat)

        img_loss = F.cross_entropy(img_out, label)

        con_loss = []
        for idx, out_t in enumerate(con_out):
            con_loss.append(F.mse_loss(out_t, table_features_con[:, idx]))
        con_loss_mean = torch.stack(con_loss, dim=0).mean(dim=0) if table_features_con.shape[1] != 0 else 0

        cat_loss = []
        for idx, out_t in enumerate(cat_out):
            cat_loss.append(F.cross_entropy(out_t, table_features_cat[:, idx]))
        cat_loss_mean = torch.stack(cat_loss, dim=0).mean(dim=0) if table_features_cat.shape[1] != 0 else 0

        loss = self.loss_weight_dict['img_loss'] * img_loss + self.loss_weight_dict['con_loss'] * con_loss_mean \
               + self.loss_weight_dict['cat_loss'] * cat_loss_mean

        table_embed_loss = F.cross_entropy(table_embed_out, label)
        loss = loss + self.loss_weight_dict['tab_loss'] * table_embed_loss

        preds = self.net_img_clf.backbone(image_features)
        preds = self.net_img_clf.img_fc(preds)
        val_acc = self.test_acc(preds, label).item()

        self.log("val_img_loss", img_loss)
        self.log("val_tab_con_loss", con_loss_mean)
        self.log("val_tab_cat_loss", cat_loss_mean)
        self.log("val_tab_embed_loss", table_embed_loss)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 5 != 0:
            return

        valid_dataset = self.valid_loader.dataset
        ok = check_model_finite(self.net_img_clf.backbone, tag="backbone-before-OT")
        # 若返回 False，说明权重/BN 已经是 NaN/Inf，需要从干净权重恢复或重训/降学习率

        self.net_img_clf.compute_OT(valid_dataset, device=self.device)
        self.net_img_clf.mask = self.net_img_clf.get_mask('res_tmp/cluster_res_' + str(self.img_reduction_dim) +
                                                          '_Adoption_Updating.txt', 'res_tmp/OToutput_' +
                                                          str(self.img_reduction_dim) + '_Adoption_Updating.txt')
        if self.reverse:
            self.net_img_clf.mask = 1 - self.net_img_clf.mask
        return

    def test_step(self, batch, batch_idx):
        # 1. 获取数据和标签 (标签可能是类别索引，也可能是连续值)
        (_, _, image_features), label = batch
        
        # 2. 模型前向传播，得到原始输出 (logits 或连续值)
        preds = self.net_img_clf.backbone(image_features)
        preds = self.net_img_clf.img_fc(preds)
        
        # --- 根据任务类型执行不同的逻辑 ---
        if self.hparams.task == 'classification':
            # 计算分类损失
            loss = F.cross_entropy(preds, label)
            # 将 Logits 转换为概率
            preds_probs = torch.softmax(preds, dim=1)
            # 更新分类指标的状态
            self.test_acc.update(preds_probs, label)
            self.test_auc.update(preds_probs, label)
            self.test_macro_f1.update(preds_probs, label)
            
        elif self.hparams.task == 'regression':
            # 回归任务的输出preds和标签label都是连续值
            # 确保label的形状和preds匹配
            if label.ndim == 1:
                label = label.unsqueeze(1) # 将 (N,) 变为 (N, 1)
            
            # 计算回归损失 (MSE)
            loss = F.mse_loss(preds, label)
            # 更新回归指标的状态
            self.test_mse.update(preds, label)
            self.test_mae.update(preds, label)
            self.test_r2.update(preds, label)

        # 记录每个批次的损失值
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs):
        # 在所有测试批次结束后，根据任务类型计算并记录最终的聚合指标
        if self.hparams.task == 'classification':
            # 计算并记录分类指标
            self.log("test_acc", self.test_acc.compute(), prog_bar=True)
            self.log("test_auc", self.test_auc.compute(), prog_bar=True)
            self.log("test_macro_f1", self.test_macro_f1.compute(), prog_bar=True)
            # 重置状态
            self.test_acc.reset()
            self.test_auc.reset()
            self.test_macro_f1.reset()

        elif self.hparams.task == 'regression':
            # 计算并记录回归指标
            # 从MSE计算RMSE
            final_mse = self.test_mse.compute()
            final_rmse = torch.sqrt(final_mse)
            
            self.log("test_rmse", final_rmse, prog_bar=True)
            self.log("test_mae", self.test_mae.compute(), prog_bar=True)
            self.log("test_r2", self.test_r2.compute(), prog_bar=True)
            # 重置状态
            self.test_mse.reset()
            self.test_mae.reset()
            self.test_r2.reset()

    # def configure_optimizers(self):
    #     optimizer = optim.SGD(self.net_img_clf.parameters(), lr=1e-3, weight_decay=1e-5, momentum=0.1)
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.5)
    #     optimizer_config = {
    #         "optimizer": optimizer,
    #     }
    #     print("optimizer_config:\n", optimizer_config)
    #     if scheduler:
    #         optimizer_config.update({
    #             "lr_scheduler": {
    #                 "name": 'MultiStep_LR_scheduler',
    #                 "scheduler": scheduler,
    #             }})
    #         print("scheduler_config:\n", scheduler.state_dict())
    #     return optimizer_config

    def configure_optimizers(self):
        # --- 原来的代码 (注释掉) ---
        # optimizer = optim.SGD(self.net_img_clf.parameters(), lr=1e-3, weight_decay=1e-5, momentum=0.1)
        
        # +++ 修改后的代码 +++
        # 换用 AdamW 优化器，它能更好地处理多任务产生的复杂梯度，更加稳定。
        # 初始学习率调整为 1e-4，这是一个更安全、更常用的起点。
        # weight_decay 也可适当调整，1e-4 是一个常见值。
        optimizer = optim.AdamW(self.net_img_clf.parameters(), lr=self.hparams.optimizer.lr , weight_decay=1e-4)
        # 学习率调度器的逻辑保持不变，它会作用于新的优化器
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.5)
        
        optimizer_config = {
            "optimizer": optimizer,
        }
        
        print("optimizer_config:\n", optimizer_config)
        
        if scheduler:
            optimizer_config.update({
                "lr_scheduler": {
                    "name": 'MultiStep_LR_scheduler',
                    "scheduler": scheduler,
                    "interval": "epoch", # 明确调度器在每个epoch后更新
                    "frequency": 1,
                }
            })
            print("scheduler_config:\n", scheduler.state_dict())
            
        return optimizer_config
