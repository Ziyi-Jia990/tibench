import os 
import sys
import time
import random
from multiprocessing import Queue
import shutil # <--- 新增导入

import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from trainers.pretrain import pretrain
from trainers.evaluate import evaluate
from trainers.test import test
from trainers.generate_embeddings import generate_embeddings
from utils.utils import grab_arg_from_checkpoint, prepend_paths, re_prepend_paths

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

import wandb
wandb.init(mode="disabled")


#@hydra.main(config_path='./configs', config_name='config', version_base=None)
def run(args: DictConfig):
  pl.seed_everything(args.seed)
  args = prepend_paths(args)
  time.sleep(random.randint(1,5)) # Prevents multiple runs getting the same version when launching many jobs at once

  output_filename = args.output_filename
  with open(output_filename, "a") as f:
      f.write(f"Target: {args.target}, Batch Size: {args.batch_size}, Learning Rate: {args.optimizer.lr}\n")


  if args.resume_training:
    if args.wandb_id:
      wandb_id = args.wandb_id
    checkpoint = args.checkpoint
    ckpt = torch.load(args.checkpoint)
    args = ckpt['hyper_parameters']
    args = OmegaConf.create(args)
    #with open_dict(args):
    args.checkpoint = checkpoint
    args.resume_training = True
    if not 'wandb_id' in args or not args.wandb_id:
      args.wandb_id = wandb_id
    # Run prepend again in case we move to another server and need to redo the paths
    args = re_prepend_paths(args)
  
  if args.generate_embeddings:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args, 'dataset')
    generate_embeddings(args)
    return args
  
  base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
  if args.use_wandb:
    if args.resume_training and args.wandb_id:
      wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, save_dir=base_dir, offline=args.offline, id=args.wandb_id, resume='must')
    else:
      wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, save_dir=base_dir, offline=args.offline, log_model='all')
  else:
    wandb_logger = WandbLogger(name=args.target, project='Test', entity='', save_dir=base_dir, offline=args.offline)
  args.wandb_id = wandb_logger.version

  if args.checkpoint and not args.resume_training:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args, 'datatype')
      
  if args.pretrain:
    model = pretrain(args, wandb_logger)
    args.checkpoint = os.path.join(base_dir, 'runs', args.datatype, wandb_logger.experiment.name, f'checkpoint_last_epoch_{args.max_epochs-1:02}.ckpt')
  
  if args.test:
    test(args, wandb_logger, model)
  elif args.evaluate:
    evaluate(args, wandb_logger)

  # --- vvvv 在这里添加或替换为下面的【多目录清理】代码 vvvv ---
  
  print(f"\n--- 任务流程结束，开始清理指定目录 ---")

  # 1. 定义一个要删除的目录列表
  #    将所有需要固定删除的路径都放在这里
  directories_to_delete = [
      '/data0/jiazy/tibench/outputs'
  ]

  # 2. 从配置中获取 checkpoint 目录，如果有效，也添加到列表中
  checkpoint_dir_from_args = args.checkpoint_dir
  if checkpoint_dir_from_args and isinstance(checkpoint_dir_from_args, str):
      directories_to_delete.append(checkpoint_dir_from_args)
  else:
      print("ℹ️ 未在配置中指定 'checkpoint_dir' 或其值无效，跳过 checkpoint 目录的清理。")

  # 3. 遍历列表，依次删除每个目录
  for dir_path in directories_to_delete:
      print(f"\n-> 正在处理目标: {dir_path}")
      
      # 再次检查路径是否有效
      if not dir_path or not isinstance(dir_path, str):
          print(f"ℹ️ 无效的路径，跳过。")
          continue

      if os.path.isdir(dir_path):
          try:
              shutil.rmtree(dir_path)
              print(f"✅ 成功删除目录: {dir_path}")
          except OSError as e:
              # 打印更详细的错误信息
              print(f"❌ 删除目录 '{dir_path}' 时发生错误: {e}")
      else:
          print(f"ℹ️ 目标不是一个有效目录或不存在，无需删除: {dir_path}")
          
  # --- ^^^^ 添加代码结束 ^^^^ ---

  wandb.finish()
  del wandb_logger

@property
def exception(self):
  if self._pconn.poll():
    self._exception = self._pconn.recv()
  return self._exception

@hydra.main(config_path='./configs', config_name='config', version_base=None)
def control(args: DictConfig):
  run(args)

if __name__ == "__main__":
  control()

