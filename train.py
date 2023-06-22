from omegaconf import OmegaConf
import torch.multiprocessing as mp
from trainer import Trainer


def main_worker(gpu, cfg):
    trainer = Trainer(cfg=cfg, rank=gpu, gpu=gpu)
    trainer.sequential_training()


if __name__ == '__main__':
    cfg = OmegaConf.load("config.yaml")
    mp.spawn(main_worker, nprocs=cfg.GPUS, args=(cfg,))
