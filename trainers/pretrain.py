import os 
import sys

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.utils import grab_image_augmentations, grab_wids, create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset, \
    ContrastiveImagingAndTabularDataset_PetFinder

from datasets.CHARMS_dataset import PetFinderConCatImageDataset
from datasets.CHARMS_new_dataset import ConCatImageDataset


from datasets.ContrastiveImageDataset import ContrastiveImageDataset
from datasets.ContrastiveTabularDataset import ContrastiveTabularDataset

from models.MultimodalSimCLR import MultimodalSimCLR
# from models.CHARMS import ImageModelPetFinderWithRTDL, DVMCharmsFinetuner
from models.CHARMS_Model import ImageModelWithRTDL
from models.SimCLR import SimCLR
from models.SwAV_Bolt import SwAV
from models.BYOL_Bolt import BYOL
from models.SimSiam_Bolt import SimSiam
from models.BarlowTwins import BarlowTwins
from models.SCARF import SCARF


def load_datasets(hparams):
  if hparams.datatype == 'multimodal':
    transform = grab_image_augmentations(hparams.img_size, hparams.target)
    hparams.transform = transform.__repr__()  # 一个魔法方法，用于返回一个对象的“官方”字符串表示形式
    if hparams.target == 'adoption':
      train_dataset = ContrastiveImagingAndTabularDataset_PetFinder(
        hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate,
        hparams.data_train_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_train, hparams.img_size, hparams.live_loading)
      val_dataset = ContrastiveImagingAndTabularDataset_PetFinder(
        hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate,
        hparams.data_val_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_val, hparams.img_size, hparams.live_loading)
      hparams.input_size = train_dataset.get_input_size()
      return train_dataset, val_dataset
    else:
      train_dataset = ContrastiveImagingAndTabularDataset(
        hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_train_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_train, hparams.img_size, hparams.live_loading)
      val_dataset = ContrastiveImagingAndTabularDataset(
        hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_val_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_val, hparams.img_size, hparams.live_loading)
      hparams.input_size = train_dataset.get_input_size()
  elif hparams.datatype == 'charms':
    print(f"Loading {hparams.target} training data...")
    train_dataset = ConCatImageDataset(
          tabular_csv_path=hparams.data_train_tabular,        # train_features.csv
          image_paths_pt=hparams.data_train_imaging,          # train_paths.pt
          label_pt=hparams.labels_train,                      # train_labels.pt
          field_lengths_path=hparams.field_lengths_tabular,   # tabular_lengths.pt
          target=hparams.target,
          train=True,
          task="classification"  # 或 "regression"
      )

    valid_dataset = ConCatImageDataset(
        tabular_csv_path=hparams.data_val_tabular,          # val_features.csv
        image_paths_pt=hparams.data_val_imaging,            # val_paths.pt
        label_pt=hparams.labels_val,                        # val_labels.pt
        field_lengths_path=hparams.field_lengths_tabular,   # 同一份 lengths
        target=hparams.target,
        train=False,
        task="classification"  # 或 "regression"
    )
    hparams.input_size = train_dataset.__len__()
    return train_dataset, valid_dataset

  elif hparams.datatype == 'imaging':
    transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.crop_scale_lower)
    hparams.transform = transform.__repr__()
    if hparams.target == 'ImageNet':
      wids_0 = grab_wids(hparams.majority_class)
      wids_1 = grab_wids(hparams.minority_class)
      pass
    else:
      train_dataset = ContrastiveImageDataset(
        data=hparams.data_train_imaging, labels=hparams.labels_train, 
        transform=transform, delete_segmentation=hparams.delete_segmentation, 
        augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading)
      val_dataset = ContrastiveImageDataset(
        data=hparams.data_val_imaging, labels=hparams.labels_val, 
        transform=transform, delete_segmentation=hparams.delete_segmentation, 
        augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading)
  elif hparams.datatype == 'tabular':
    train_dataset = ContrastiveTabularDataset(hparams.data_train_tabular, hparams.labels_train, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    val_dataset = ContrastiveTabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return train_dataset, val_dataset


def select_model(hparams, train_dataset):
  if hparams.datatype == 'multimodal':
    model = MultimodalSimCLR(hparams)
  elif hparams.datatype == 'charms':
    n_num_features = train_dataset.get_num_continuous()
    cat_cardinalities = train_dataset.get_cat_cardinalities()

    # 3) Model
    model = ImageModelWithRTDL(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        num_classes=hparams.num_classes,
        target=hparams.target,
        img_reduction_dim=getattr(hparams, "img_reduction_dim", 40),
        ot_update_every=getattr(hparams, "ot_update_every", 5),
        ot_dir=getattr(hparams, "ot_dir", "res_tmp"),
        backbone_name=getattr(hparams, "backbone_name", "resnet"),
    )

    
  elif hparams.datatype == 'imaging':
    if hparams.loss.lower() == 'byol':
      model = BYOL(**hparams)
    elif hparams.loss.lower() == 'simsiam':
      model = SimSiam(**hparams)
    elif hparams.loss.lower() == 'swav':
      if not hparams.resume_training:
        model = SwAV(gpus=1, nmb_crops=(2,0), num_samples=len(train_dataset),  **hparams)
      else:
        model = SwAV(**hparams)
    elif hparams.loss.lower() == 'barlowtwins':
      model = BarlowTwins(**hparams)
    else:
      model = SimCLR(hparams)
  elif hparams.datatype == 'tabular':
    model = SCARF(hparams)
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return model


def pretrain(hparams, wandb_logger):
  """
  Train code for pretraining or supervised models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)

  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams)
  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)

  # Create logdir based on WandB run name
  # logdir = create_logdir(hparams.datatype, hparams.resume_training, wandb_logger)
  logdir = create_logdir(hparams, wandb_logger)
  
  model = select_model(hparams, train_dataset)
  
  callbacks = []

  if hparams.datatype != 'charms':
    if hparams.online_mlp:
      model.hparams.classifier_freq = float('Inf')
      callbacks.append(SSLOnlineEvaluator(z_dim = model.pooled_dim, hidden_dim = hparams.embedding_dim, num_classes = hparams.num_classes, swav = False, multimodal = (hparams.datatype=='multimodal')))
    callbacks.append(ModelCheckpoint(filename='checkpoint_last_epoch_{epoch:02d}', dirpath=logdir, save_on_train_epoch_end=True, auto_insert_metric_name=False))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  else:
    checkpoint_callback = ModelCheckpoint(
    filename='checkpoint_last_{epoch:02d}_{val_acc:.4f}',
    dirpath=logdir, save_last=True, save_weights_only=False,
    monitor="val_acc", mode="max", save_top_k=5
    )
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_acc',  patience=5, mode='max')
    callbacks = [checkpoint_callback, early_stopping]

  # trainer = Trainer.from_argparse_args(hparams, gpus=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, enable_progress_bar=hparams.enable_progress_bar)
  trainer = Trainer.from_argparse_args(
      hparams,
      gpus=1,
      callbacks=callbacks,
      logger=wandb_logger,
      max_epochs=hparams.max_epochs,
      check_val_every_n_epoch=hparams.check_val_every_n_epoch,
      limit_train_batches=hparams.limit_train_batches,
      limit_val_batches=hparams.limit_val_batches,
      enable_progress_bar=hparams.enable_progress_bar,
      accumulate_grad_batches=hparams.accumulate_grad_batches,         
      precision=32,                     
      gradient_clip_val=1.0,
  )

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)
  
  return model