import os
import argparse
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.loggers import TensorBoardLogger

from configs.config import cfg
from datasets import call_load_dataset
from model import Model
from utils.eval_utils import AverageMeter, get_prompts, validate
from utils.tools import copy_model, create_csv
from model import Model
from sam_lora import LoRA_Sam


def multi_main(cfg):
    prompts = ["box", "point"]
    for prompt in prompts:
        cfg.prompt = prompt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box, ckpt: str = None) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)
    # 初始化分布式环境
    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_head=cfg.csv_keys)
    # 构建带LoRA的SAM模型
    with fabric.device:
        model = Model(cfg)
        model.setup()
        LoRA_Sam(model.model, 4)
    # 加载验证数据集
    load_datasets = call_load_dataset(cfg)
    _, val_data = load_datasets(cfg, model.model.image_encoder.img_size)

    fabric.print(f"Val Data: {len(val_data) * cfg.val_batchsize}")
    val_data = fabric._setup_dataloader(val_data)
    # 加载模型检查点
    if ckpt is not None:
        full_checkpoint = fabric.load(ckpt)
        model.load_state_dict(full_checkpoint["model"])

    validate(fabric, cfg, model, val_data, name=cfg.name, iters=0)

    del model, val_data


if __name__ == "__main__":
    torch.cuda.empty_cache() # 清理GPU缓存
    torch.set_float32_matmul_precision('high') # 设置矩阵运算精度
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids # 指定使用的GPU

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str) # 接收模型检查点路径
    args = parser.parse_args()

    main(cfg, args.ckpt) # 执行单提示模式评估
    # multi_main(cfg, args.ckpt)
    torch.cuda.empty_cache()
