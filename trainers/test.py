from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.CHARMS_dataset import PetFinderConCatImageDataset
from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
# 确保其他必要的导入也包含在内

def test(hparams, wandb_logger=None, model=None):
    """
    测试训练好的模型，并将最终的三个核心指标追加写入到 results.txt 文件。
    """
    pl.seed_everything(hparams.seed)
    
    # --- 这部分数据加载逻辑保持不变 ---
    if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
        test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading)
        print(test_dataset.transform_val.__repr__())
    elif hparams.datatype == 'tabular':
        test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
        hparams.input_size = test_dataset.get_input_size()
    elif hparams.datatype == 'charms':
        if hparams.target == 'adoption':
            test_dataset = PetFinderConCatImageDataset(hparams.data_test_eval_tabular, hparams.data_test_eval_imaging)
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
    
    output_filename = "results.txt"
    string_to_write = None
    
    # 根据任务类型构建输出字符串
    if hparams.task == 'classification':
        # 检查分类所需的所有指标是否存在
        if all(k in metrics for k in ['test_acc', 'test_auc', 'test_macro_f1']):
            acc = metrics['test_acc']
            auc = metrics['test_auc']
            f1 = metrics['test_macro_f1']
            string_to_write = f"acc:{acc};auc:{auc};macro_f1:{f1}"
        else:
            print("错误: 缺少一个或多个分类指标 (test_acc, test_auc, test_macro_f1)。")

    elif hparams.task == 'regression':
        # 检查回归所需的所有指标是否存在
        if all(k in metrics for k in ['test_rmse', 'test_mae', 'test_r2']):
            rmse = metrics['test_rmse']
            mae = metrics['test_mae']
            r2 = metrics['test_r2']
            string_to_write = f"rmse:{rmse};mae:{mae};r2:{r2}"
        else:
            print("错误: 缺少一个或多个回归指标 (test_rmse, test_mae, test_r2)。")

    else:
        print(f"错误: 不支持的任务类型 '{hparams.task}'。")

    # 如果成功构建了字符串，就将其追加写入文件
    if string_to_write:
        try:
            # 使用模式 "a" (append) 来追加内容
            with open(output_filename, "a") as f:
                f.write(string_to_write + "\n")
            print(f"指标已成功追加到文件: {output_filename}")
        except Exception as e:
            print(f"错误：写入文件 {output_filename} 失败。原因: {e}")
    else:
        print(f"文件 '{output_filename}' 未被写入，因为未能成功构建指标字符串。")