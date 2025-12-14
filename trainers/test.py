from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.CHARMS_dataset import PetFinderConCatImageDataset
from datasets.CHARMS_new_dataset import ConCatImageDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint

# 确保其他必要的导入也包含在内

def test(hparams, wandb_logger=None, model=None):
    """
    测试训练好的模型，并将最终的三个核心指标追加写入到 results.txt 文件。
    """
    pl.seed_everything(hparams.seed)
    
    # --- 这部分数据加载逻辑 ---
    if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
        test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), task=hparams.task, target=hparams.target, train=False, live_loading=hparams.live_loading)
        print(test_dataset.transform_val.__repr__())
    elif hparams.datatype == 'tabular':
        test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
        hparams.input_size = test_dataset.get_input_size()
    elif hparams.datatype == 'imaging_and_tabular':
        print("[INFO] Datatype is 'imaging_and_tabular', loading PetFinderConCatImageDataset.")
        test_dataset = PetFinderConCatImageDataset(hparams.data_test_eval_tabular, hparams.data_test_eval_imaging)
        hparams.input_size = test_dataset.__len__()
    elif hparams.datatype == 'charms':
        print(f"Loading {hparams.target} test data using UnifiedSupervisedDataset...")
        
        # [Fix] 动态获取 task，不再硬编码
        task_type = getattr(hparams, "task", "classification")

        test_dataset = ConCatImageDataset(
            tabular_csv_path=hparams.data_test_eval_tabular,     
            image_paths_pt=hparams.data_test_eval_imaging,       
            label_pt=hparams.labels_test_eval_imaging,                      
            field_lengths_path=hparams.field_lengths_tabular,   # tabular_lengths.pt
            target=hparams.target,
            train=False,
            task=task_type  # <--- 传入动态 task
        )
        hparams.input_size = test_dataset.__len__()
    else:
        raise Exception('argument dataset must be set to imaging, tabular or multimodal')

    if hparams.datatype != 'charms':
        drop = ((len(test_dataset) % hparams.batch_size) == 1)
        test_loader = DataLoader(
            test_dataset,
            num_workers=hparams.num_workers, batch_size=hparams.batch_size,
            pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)
        hparams.dataset_length = len(test_loader)
        model = Evaluator(hparams)
    else:
        test_loader = DataLoader(
            test_dataset,
            num_workers=hparams.num_workers, batch_size=hparams.batch_size, shuffle=False)
        # model 已经在 run.py 中通过 pretrain() 返回或者是从 checkpoint 加载的
        model = model

    model.freeze()
    trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
    print("test starts")
    print(test_dataset.__len__())
    
    results = trainer.test(model=model, dataloaders=test_loader, verbose=False) 
    print("test ends")

    # --- 核心修改：根据任务类型格式化结果并追加写入文件 ---
    if not results:
        print("警告: trainer.test() 没有返回任何结果。")
        return # 提前退出

    metrics = results[0]
    print("测试指标:", metrics)
    
    output_filename = hparams.output_filename 
    # --- 新增逻辑：直接将 metrics 字典转换为字符串并追加写入文件 ---
    try:
        # 使用模式 "a" (append) 来追加内容
        with open(output_filename, "a") as f:
            # 将 metrics 字典转换为字符串并写入
            # 由于我们在 CHARMS_Model.py 里已经把 test_rmse/mae/r2 放入 log 了，
            # 这里会自动打印出类似 {'test_rmse': 0.123, 'test_mae': 0.05, 'test_r2': 0.85} 的内容
            f.write(str(metrics) + "\n") 
        print(f"指标已成功追加到文件: {output_filename}")
    except Exception as e:
        print(f"错误：写入文件 {output_filename} 失败。原因: {e}")