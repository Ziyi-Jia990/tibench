import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.cluster import KMeans
import ot
import rtdl
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
import torch.nn.functional as F
import torch
from torchmetrics.classification import AUROC, F1Score
from torchmetrics import Accuracy


class Identity(nn.Module):
    def forward(self, x):
        return x


class ImageClassifier(nn.Module):
    def __init__(
        self,
        img_reduction_dim: int,
        model_name: str = "resnet",
        out_dims: int = 2,                 # <-- 由 config.num_classes 决定
        n_num_features: int = 0,
        cat_cardinalities: list = None,
        d_token: int = 8,
        ot_dir: str = "res_tmp",           # <-- 可配置
        ot_tag: str = "dataset",           # <-- 用 target 来标记
    ):
        super().__init__()
        cat_cardinalities = cat_cardinalities or []

        self.model_name = model_name
        self.img_reduction_dim = img_reduction_dim
        self.table_dim = n_num_features + len(cat_cardinalities)

        self.ot_dir = ot_dir
        self.ot_tag = ot_tag
        os.makedirs(self.ot_dir, exist_ok=True)

        # ---- backbone + img head ----
        if model_name == "resnet":
            backbone = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
            in_dims = backbone.fc.in_features
            img_fc = nn.Sequential(nn.Linear(in_dims, 1024), nn.Linear(1024, out_dims))
            backbone.fc = Identity()

        elif model_name == "densenet":
            backbone = models.densenet121(weights=models.densenet.DenseNet121_Weights.IMAGENET1K_V1)
            in_dims = backbone.classifier.in_features
            img_fc = nn.Sequential(nn.Linear(in_dims, 1024), nn.Linear(1024, out_dims))
            backbone.classifier = Identity()

        elif model_name == "inception":
            backbone = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
            in_dims = backbone.fc.in_features
            img_fc = nn.Sequential(nn.Linear(in_dims, 1024), nn.Linear(1024, out_dims))
            backbone.fc = Identity()

        elif model_name == "mobilenet":
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_dims = backbone.classifier[-1].in_features
            img_fc = nn.Sequential(nn.Linear(in_dims, 1024), nn.Linear(1024, out_dims))
            backbone.classifier[-1] = Identity()

        else:
            raise ValueError(f"Unsupported model_name={model_name}")

        self.in_dims = in_dims
        self.backbone = backbone
        self.img_fc = img_fc

        # con_fc
        self.con_fc = nn.ModuleList([nn.Linear(in_dims, 1) for _ in range(n_num_features)])
        self.con_fc_num = len(self.con_fc)

        # cat_fc
        self.cat_fc = nn.ModuleList([nn.Linear(in_dims, int(c)) for c in cat_cardinalities])
        self.cat_fc_num = len(self.cat_fc)

        # tabular model
        self.tab_model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            attention_dropout=0.1,
            n_blocks=2,
            ffn_d_hidden=6,
            ffn_dropout=0.2,
            residual_dropout=0.0,
            d_out=out_dims,
        )

        self.mask = torch.ones((self.table_dim, in_dims), dtype=torch.long)

    # ---------- OT filename helpers ----------
    def _cluster_path(self) -> str:
        return os.path.join(self.ot_dir, f"cluster_res_{self.img_reduction_dim}_{self.ot_tag}.txt")

    def _ot_path(self) -> str:
        return os.path.join(self.ot_dir, f"OToutput_{self.img_reduction_dim}_{self.ot_tag}.txt")

    def _cluster_center_path(self) -> str:
        return os.path.join(self.ot_dir, f"cluster_centering_{self.img_reduction_dim}_{self.ot_tag}.txt")

    # ---------- forward ----------
    def forward(self, img, tab_con, tab_cat):
        mask = self.mask.to(img.device)
        extracted_feats = self.backbone(img)
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

        _, table_embed_out = self.tab_model(tab_con, tab_cat)
        return img_out, con_out, cat_out, table_embed_out

    # ---------- OT pipeline ----------
    def compute_OT(self, dataset, device):
        test_table_feat, test_channel_feat = self.getTableChannelFeat(dataset, device)
        cost = self.getCostMatrix(test_table_feat, test_channel_feat)
        P, _ = self.compute_coupling(test_table_feat, test_channel_feat, cost)
        np.savetxt(self._ot_path(), P)
        return P

    def get_mask(self, cluster_path: str = None, OT_path: str = None):
        cluster_path = cluster_path or self._cluster_path()
        OT_path = OT_path or self._ot_path()

        cluster_dict = {}
        with open(cluster_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = list(line[1:-2].split(","))
                cluster_dict[idx] = np.array(line, dtype=int)

        OT = np.loadtxt(OT_path)

        mask = np.zeros((self.table_dim, self.in_dims))
        for i in range(self.table_dim):
            for idx_OT, j in enumerate(OT[i]):
                channel_id = cluster_dict[idx_OT]
                mask[i, channel_id] = 1 if j != 0 else 0
        return torch.tensor(mask, dtype=torch.long)

    def getTableChannelFeat(self, dataset, device):
        test_channel_feat = []
        test_table_feat = []

        for row in dataset:
            feats, _ = row
            table_features_con, table_features_cat, image = feats[0], feats[1], feats[2]

            table_features_con = table_features_con.unsqueeze(0).to(device)
            table_features_cat = table_features_cat.unsqueeze(0).to(device)
            image = image.unsqueeze(0).to(device)

            channel_feat = self.getChannelFeature(image)
            table_feat = self.getTableFeature(table_features_con, table_features_cat)

            test_channel_feat.append(channel_feat.unsqueeze(1))
            test_table_feat.append(table_feat.unsqueeze(1))

        test_channel_feat = torch.cat(test_channel_feat, dim=1)
        test_table_feat = torch.cat(test_table_feat, dim=1)
        return test_table_feat, test_channel_feat

    def getChannelFeature(self, image):
        self.backbone.eval()
        if self.model_name == "mobilenet":
            new_net = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            new_net = nn.Sequential(*list(self.backbone.children())[:-2])

        channel_feat = new_net(image)                 # [1, C, 7, 7]
        channel_feat = channel_feat.squeeze(0)        # [C, 7, 7]
        channel_feat = channel_feat.reshape((self.in_dims, -1)).detach().cpu().numpy()
        return torch.tensor(channel_feat, dtype=torch.float)  # (C, 49)

    def getTableFeature(self, table_features_con, table_features_cat):
        self.tab_model.eval()
        if self.con_fc_num == 0:
            table_features_con = None
        if self.cat_fc_num == 0:
            table_features_cat = None
        table_features_embed, _ = self.tab_model(table_features_con, table_features_cat)
        return table_features_embed.squeeze(0)

    def getCostMatrix(self, test_table_feat, test_channel_feat):
        src_x = test_table_feat.detach().cpu().numpy()
        tar_x = test_channel_feat.detach().cpu().numpy()

        img_embed = tar_x.shape[2]
        tar_x_flat = tar_x.reshape((self.in_dims, -1))

        kmeans = KMeans(n_clusters=self.img_reduction_dim, random_state=0, n_init="auto").fit(tar_x_flat)

        # save cluster centers
        tar_centers = kmeans.cluster_centers_.reshape((self.img_reduction_dim, -1, img_embed))
        with open(self._cluster_center_path(), "w") as f:
            for i in range(self.img_reduction_dim):
                f.write(str(tar_centers[i]) + "\n")

        # save assignments (needed by get_mask)
        labels = kmeans.labels_
        with open(self._cluster_path(), "w") as f:
            for i in range(self.img_reduction_dim):
                f.write(str(np.where(labels == i)[0].tolist()) + "\n")

        cost = np.zeros((src_x.shape[0], tar_centers.shape[0]))
        for i in range(src_x.shape[0]):
            src_sim = src_x[i] / np.linalg.norm(src_x[i])
            src_sim = np.dot(src_sim, src_sim.transpose(1, 0))
            for j in range(tar_centers.shape[0]):
                tar_sim = tar_centers[j] / np.linalg.norm(tar_centers[j])
                tar_sim = np.dot(tar_sim, tar_sim.transpose(1, 0))
                cost[i, j] = ((src_sim - tar_sim) ** 2).sum()
        return cost

    def compute_coupling(self, X_src, X_tar, Cost):
        P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(self.img_reduction_dim), Cost, numItermax=100000)
        W = np.sum(P * np.array(Cost))
        return P, W




class ImageModelWithRTDL(pl.LightningModule):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list,
        num_classes: int,
        target: str,
        reverse: bool = False,
        img_reduction_dim: int = 40,
        ot_update_every: int = 5,
        ot_dir: str = "res_tmp",
        loss_weight_dict: dict = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        momentum: float = 0.1,
        milestones: list = None,
        gamma: float = 0.5,
        backbone_name: str = "resnet",
        d_token: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_weight_dict"])

        self.net_img_clf = ImageClassifier(
            model_name=backbone_name,
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            img_reduction_dim=img_reduction_dim,
            out_dims=num_classes,
            d_token=d_token,
            ot_dir=ot_dir,
            ot_tag=target,   # <-- 关键：用数据集 target 做标签
        )

        self.num_classes = num_classes
        self.target = target
        self.reverse = reverse
        self.img_reduction_dim = img_reduction_dim
        self.ot_update_every = ot_update_every

        self.loss_weight_dict = loss_weight_dict or {
            "con_loss": 0.03,
            "cat_loss": 0.03,
            "tab_loss": 0.6,
            "img_loss": 1.0,
        }

        # ===== metrics (macro) =====
        self.val_acc_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auc_metric = AUROC(task="multiclass", num_classes=num_classes)
        self.val_macro_f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # 用于 test：同样一套（保持 PetFinder 习惯：test_step update, test_epoch_end compute）
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_macro_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # 如果你也想在 val 里算同样指标，也可以加：


        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.milestones = milestones or [5, 10, 15, 20]
        self.gamma = gamma

    def _compute_losses(self, batch):
        (table_features_con, table_features_cat, image_features), label = batch

        img_out, con_out, cat_out, table_embed_out = self.net_img_clf(
            image_features, table_features_con, table_features_cat
        )

        img_loss = F.cross_entropy(img_out, label)
        tab_embed_loss = F.cross_entropy(table_embed_out, label)

        if table_features_con is not None and table_features_con.numel() > 0 and table_features_con.shape[1] > 0:
            con_loss = torch.stack([
                F.mse_loss(out_t, table_features_con[:, idx])
                for idx, out_t in enumerate(con_out)
            ]).mean()
        else:
            con_loss = torch.tensor(0.0, device=self.device)

        if table_features_cat is not None and table_features_cat.numel() > 0 and table_features_cat.shape[1] > 0:
            cat_loss = torch.stack([
                F.cross_entropy(out_t, table_features_cat[:, idx])
                for idx, out_t in enumerate(cat_out)
            ]).mean()
        else:
            cat_loss = torch.tensor(0.0, device=self.device)

        loss = (
            self.loss_weight_dict["img_loss"] * img_loss
            + self.loss_weight_dict["con_loss"] * con_loss
            + self.loss_weight_dict["cat_loss"] * cat_loss
            + self.loss_weight_dict["tab_loss"] * tab_embed_loss
        )

        return loss, img_loss, con_loss, cat_loss, tab_embed_loss, img_out, label

    def training_step(self, batch, batch_idx):
        loss, img_loss, con_loss, cat_loss, tab_embed_loss, _, _ = self._compute_losses(batch)

        self.log("train/img_loss", img_loss)
        self.log("train/tab_con_loss", con_loss)
        self.log("train/tab_cat_loss", cat_loss)
        self.log("train/tab_embed_loss", tab_embed_loss)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, img_loss, con_loss, cat_loss, tab_embed_loss, img_out, label = self._compute_losses(batch)

        # ===== metrics: image head only (match PetFinder) =====
        probs = torch.softmax(img_out, dim=1)
        preds = torch.argmax(img_out, dim=1)

        val_acc = self.val_acc_metric(preds, label)
        val_auc = self.val_auc_metric(probs, label)
        val_f1  = self.val_macro_f1_metric(preds, label)

        # 你原来的 logs（保留）
        self.log("val/img_loss", img_loss, on_step=False, on_epoch=True)
        self.log("val/tab_con_loss", con_loss, on_step=False, on_epoch=True)
        self.log("val/tab_cat_loss", cat_loss, on_step=False, on_epoch=True)
        self.log("val/tab_embed_loss", tab_embed_loss, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        # ✅ charms 训练代码监控的是 val_acc：一定要有这个 key
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # 可选：你想在日志里也看到 auc/f1（不影响 checkpoint）
        self.log("val_auc", val_auc, on_step=False, on_epoch=True)
        self.log("val_macro_f1", val_f1, on_step=False, on_epoch=True)

        return loss



    def on_validation_epoch_end(self):
        if self.ot_update_every <= 0:
            return
        if (self.current_epoch + 1) % self.ot_update_every != 0:
            return

        # 从 trainer 里拿 val dataset（不写死任何路径）
        valid_dataset = None
        try:
            vdl = self.trainer.val_dataloaders
            if isinstance(vdl, (list, tuple)) and len(vdl) > 0:
                valid_dataset = vdl[0].dataset
            else:
                valid_dataset = vdl.dataset
        except Exception:
            valid_dataset = None

        if valid_dataset is None:
            print("[RTDL] Warning: cannot access val dataset, skip OT update.")
            return

        self.net_img_clf.compute_OT(valid_dataset, device=self.device)
        self.net_img_clf.mask = self.net_img_clf.get_mask()  # 自动用 ot_dir + ot_tag

        if self.reverse:
            self.net_img_clf.mask = 1 - self.net_img_clf.mask

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.net_img_clf.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "name": "MultiStepLR"},
        }
    
    def test_step(self, batch, batch_idx):
        (_, _, image_features), label = batch

        feats = self.net_img_clf.backbone(image_features)
        logits = self.net_img_clf.img_fc(feats)

        loss = F.cross_entropy(logits, label)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        # ===== update metrics (match PetFinder) =====
        self.test_acc.update(preds, label)          # acc 用 argmax 类别
        self.test_auc.update(probs, label)          # auc 用概率
        self.test_macro_f1.update(preds, label)     # f1 用 argmax 类别

        # 记录 loss（epoch 聚合）
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_epoch_end(self, outputs):
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.log("test_auc", self.test_auc.compute(), prog_bar=True)
        self.log("test_macro_f1", self.test_macro_f1.compute(), prog_bar=True)

        self.test_acc.reset()
        self.test_auc.reset()
        self.test_macro_f1.reset()

