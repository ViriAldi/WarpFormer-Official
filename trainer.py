import os
import sys
import logging
import datetime
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from tensorboardX import SummaryWriter

from warpformer_engine import WarpFormerEngine

import dataloaders.video_transforms as tr
from dataloaders.davis_eval import DAVIS2017Eval
from dataloaders.davis_train import DAVIS2017Train
from dataloaders.ytb_train import YoutubeVOS_Train
from dataloaders.ytb_eval import YoutubeVOS_Eval
from dataloaders.mose_train import MOSE2023_Train

from losses import macro_dice_loss, topk_cross_entropy_loss

from utils.meters import AverageMeter
from utils.video import _overlay, _palette, _palette_davis, color_palette
from utils.learning import adjust_learning_rate, setup_optimizer_param_groups
from utils.utils import count_params, load_old_arch

from davis2017_evaluation.evaluation import DAVISEvaluation


cv2.setNumThreads(0)
LOGGER = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, cfg, rank=0, gpu=0):

        self.gpu = gpu
        self.rank = rank
        self.cfg = cfg

        if rank == 0:
            LOGGER.setLevel(level=logging.INFO)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(level=logging.INFO)
            LOGGER.addHandler(stream_handler)
        else:
            LOGGER.setLevel(level=logging.WARNING)

        torch.cuda.set_device(self.gpu)

        torch.backends.cudnn.benchmark = True

        self.engine = WarpFormerEngine(cfg, self.rank)

        params = setup_optimizer_param_groups(self.engine, cfg.WEIGHT_DECAY, cfg.LR)
        self.optimizer = optim.AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        LOGGER.info(f"Optimizer ready: LR={cfg.LR}; WD={cfg.WEIGHT_DECAY}")
        LOGGER.info(f"Total params: {count_params(self.engine)/1e6:.3f}M")
        LOGGER.info(f"Trainable params: {count_params(self.engine, True)/1e6:.3f}M")

        self.enable_amp = cfg.AUTOMATIC_MIXED_PRECISION
        if self.enable_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        LOGGER.info(f"AMP: {self.enable_amp}")

        dist.init_process_group(backend="nccl", init_method=cfg.DIST_URL, world_size=cfg.GPUS, rank=rank)
        self.engine = nn.SyncBatchNorm.convert_sync_batchnorm(self.engine).cuda()
        self.dist_engine = torch.nn.parallel.DistributedDataParallel(module=self.engine, device_ids=[self.gpu], output_device=self.gpu, find_unused_parameters=True, broadcast_buffers=False)

        self.pretrained = False
        self.load_old = False

        if self.pretrained:
            self.ckpt_path = "checkpoints/28-03-2023_00-07-58/WarpFormer_50000.pth"
            self.ckpt = torch.load(self.ckpt_path)
            
            if self.load_old:
                missing_keys = load_old_arch(self.engine.model, self.ckpt["state_dict"])
                if (len(missing_keys) > 0):
                    LOGGER.info(f"Unexpected missing keys: {missing_keys}")
            else:
                self.engine.load_state_dict(self.ckpt["state_dict"])
                # self.engine.load_state_dict({k[7:]: v for k, v in self.ckpt["state_dict"].items()})
                self.optimizer.load_state_dict(self.ckpt["optimizer"])

            self.scaler.load_state_dict(self.ckpt["scaler"])
            LOGGER.info(f"Loaded checkpoint: {self.enable_amp}")
        
        self.step = 0
        self.timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.eval_output_root = os.path.join(cfg.VIDEO_DIR, self.timestamp)
        LOGGER.info(f"Starting timestamp: {self.timestamp}")
        LOGGER.info(f"Starting from step: {self.step}")

        if self.step > cfg.FREEZE_ID_BANK_STEP:
            for p in self.engine.model.id_bank.parameters():
                p.requires_grad = False
    
    def build_dataloader(self):
        cfg = self.cfg
        composed_transforms = transforms.Compose([
            tr.RandomScale(min_scale=cfg.TRAIN_SCALE_MIN, max_scale=cfg.TRAIN_SCALE_MAX, short_edge=cfg.TRAIN_SHORT_EDGE_LEN),
            tr.BalancedRandomCrop(cfg.TRAIN_CROP_SIZE),
            tr.RandomColorJitter(),
            tr.RandomGrayScale(),
            tr.RandomGaussianBlur(),
            tr.RandomHorizontalFlip(prob=0.5),
            tr.Padding(cfg.TRAIN_CROP_SIZE),
            tr.ToTensor(),
        ])

        if self.step < cfg.SEQUENTIAL_TRAINING_START:
            seq_len = 5
            davis_gap = cfg.DAVIS_GAP
            ytb_gap = cfg.YTB_GAP
            mose_gap = cfg.MOSE_GAP
        else:
            seq_len = cfg.TRAIN_SEQ_LEN
            davis_gap = cfg.DAVIS_GAP_SEQUENTIAL
            ytb_gap = cfg.YTB_GAP_SEQUENTIAL
            mose_gap = cfg.MOSE_GAP_SEQUENTIAL

        k_ytb_start = 0.5
        k_ytb_end = 0.25
        repeats_davis = 5
        repeats_ytb = k_ytb_start - (k_ytb_start - k_ytb_end) * (self.step / cfg.TOTAL_OPTIMIZATION_STEPS)
        repeats_mose = 1 - repeats_ytb

        dataset_davis = DAVIS2017Train(
            root=cfg.DAVIS_ROOT, 
            transform=composed_transforms,
            seq_len=seq_len, 
            rand_gap=davis_gap, 
            repeat_time=repeats_davis, 
            splits=cfg.DAVIS_TRAIN_SPLITS,
            merge_prob=cfg.DAVIS_MERGE_PROB
            )
        dataset_ytb = YoutubeVOS_Train(
            root=cfg.YTB_ROOT,
            transform=composed_transforms,
            seq_len=seq_len,
            rand_gap=ytb_gap,
            repeat_time=repeats_ytb,
            merge_prob=cfg.YTB_MERGE_PROB,
        )
        dataset_mose = MOSE2023_Train(
            root=cfg.MOSE_ROOT,
            transform=composed_transforms,
            seq_len=seq_len,
            repeat_time=repeats_mose,
            rand_gap=mose_gap,
        )
        dataset = torch.utils.data.ConcatDataset([dataset_davis, dataset_ytb, dataset_mose])

        return DataLoader(dataset,
                        batch_size=cfg.BATCH_SIZE // cfg.GPUS,
                        shuffle=True,
                        num_workers=cfg.DATA_WORKERS,
                        pin_memory=True,
                        drop_last=True,
                        prefetch_factor=cfg.PREFETCH_FACTOR,
                        persistent_workers=False,
                        )

    def calculate_loss(self, output_masks, masks):

        cfg = self.cfg

        # skip ref and prev
        masks = masks[2:]
        assert len(output_masks) == len(masks)

        n = len(masks)
        k = cfg.CE_LOSS_K_VALUE + (1 - cfg.CE_LOSS_K_VALUE) * (1 - min(1, self.step / cfg.CE_LOSS_K_START))

        dice_loss = 0
        cross_entropy_loss = 0
        for pred, gt in zip(output_masks, masks):
            dice_loss += macro_dice_loss(pred, gt) / n
            cross_entropy_loss += topk_cross_entropy_loss(pred, gt, k=k) / n

        seq_loss = cfg.LOSS_DICE_COEF * dice_loss + cfg.LOSS_CE_COEF * cross_entropy_loss
        return seq_loss, dice_loss, cross_entropy_loss

    def preprocess_masks(self, masks):
        output_masks = []
        for mask in masks:
            output_mask = mask * (mask < self.cfg.NUM_CLASSES)
            output_masks.append(output_mask)
        return output_masks

    def sequential_training(self):

        cfg = self.cfg

        # if self.rank == 0:
        #     self.eval_davis(split="val")
        #     # self.eval_ytb()

        # exit(0)

        LOGGER.info('Start training')
        self.engine.train()

        self.vos_loss = AverageMeter()
        self.dice_loss = AverageMeter()
        self.ce_loss = AverageMeter()
        self.num_objs = AverageMeter()
        self.time_start = datetime.datetime.now()

        train_end = False
        while not train_end:
            dataloader = self.build_dataloader()
            for sample in dataloader:

                if self.step == cfg.FREEZE_ID_BANK_STEP:
                    for p in self.engine.model.id_bank.parameters():
                        p.requires_grad = False

                self.curr_lr = adjust_learning_rate(
                    optimizer=self.optimizer, 
                    base_lr=cfg.LR,
                    min_lr=cfg.LR_MIN,
                    step=self.step,
                    total_steps=cfg.TOTAL_OPTIMIZATION_STEPS,
                    warm_up_steps=cfg.LR_WARMUP_STEPS,
                    lr_decay=cfg.LR_DECAY_RATE,
                )

                images = [image.cuda(self.gpu) for image in sample['images']]
                masks = [mask.cuda(self.gpu) for mask in sample['labels']]
                obj_nums = sample['obj_num'].cuda()

                masks = self.preprocess_masks(masks)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=True):
                    if self.step < cfg.SEQUENTIAL_TRAINING_START:
                        output_masks = self.dist_engine(images, masks, obj_nums, seq_train=False)
                    else:
                        output_masks = self.dist_engine(images, masks, obj_nums, seq_train=True)
                    loss, dice_loss, ce_loss = self.calculate_loss(output_masks, masks)

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.dist_engine.parameters(), cfg.CLIP_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                with torch.no_grad():
                    dist.all_reduce(loss)
                    dist.all_reduce(dice_loss)
                    dist.all_reduce(ce_loss)
                    dist.all_reduce(obj_nums)
                    loss /= cfg.GPUS
                    dice_loss /= cfg.GPUS
                    ce_loss /= cfg.GPUS
                    obj_nums = obj_nums.float().mean() / cfg.GPUS
                    if self.rank == 0:
                        self.vos_loss.update(loss)
                        self.dice_loss.update(dice_loss)
                        self.ce_loss.update(ce_loss)
                        self.num_objs.update(obj_nums)

                if self.rank == 0:
                    if self.step % cfg.TENSORBOARD_STEP == 0:
                        self.log_tensorboard()
                    if self.step % cfg.STDOUT_STEP == 0:
                        self.log_stdout()

                self.step += 1

                if self.rank == 0:
                    if self.step % cfg.CKPT_STEP == 0:
                        os.makedirs(os.path.join(cfg.CKPT_DIR, self.timestamp), exist_ok=True)
                        path_to_checkpoint = os.path.join(cfg.CKPT_DIR, self.timestamp, f"WarpFormer_{self.step}.pth")
                        ckpt = {
                            "state_dict": self.engine.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scaler": self.scaler.state_dict(),
                        }
                        torch.save(ckpt, path_to_checkpoint)

                    if self.step % cfg.DAVIS_EVAL_TRAIN_STEP == 0:
                        self.eval_davis(split="train")
                    if self.step % cfg.DAVIS_EVAL_VAL_STEP == 0:
                        self.eval_davis(split="val")
                    if self.step % cfg.DAVIS_EVAL_TESTDEV_STEP == 0:
                        self.eval_davis(split="test-dev")
                    if self.step % cfg.YTB_EVAL_STEP == 0:
                        self.eval_ytb()

                if self.step == cfg.TOTAL_OPTIMIZATION_STEPS:
                    train_end = True
                    break
            
        LOGGER.info('Stop training!')

    def log_stdout(self):
        elapsed_sec = (datetime.datetime.now() - self.time_start).seconds
        hours, minutes, seconds = elapsed_sec // 3600, (elapsed_sec % 3600) // 60, elapsed_sec % 60
        LOGGER.info(f"Step {self.step}; Time elapsed: {hours}h {minutes}m {seconds}s")
        LOGGER.info(f"VOS Loss : {self.vos_loss.moving_avg:.3f}; Dice Loss : {self.dice_loss.moving_avg:.3f}; CE Loss : {self.ce_loss.moving_avg:.3f}; #obj avg: {self.num_objs.moving_avg:.3f}; LR: {self.curr_lr:.6f}")

    def log_tensorboard(self):
        cfg = self.cfg
        if not os.path.exists(os.path.join(cfg.TENSORBOARD_DIR, self.timestamp)):
            os.makedirs(os.path.join(cfg.TENSORBOARD_DIR, self.timestamp))
            self.tblogger = SummaryWriter(os.path.join(cfg.TENSORBOARD_DIR, self.timestamp))
        self.tblogger.add_scalar("vos_loss", self.vos_loss.val, self.step)
        self.tblogger.add_scalar("dice_loss", self.dice_loss.val, self.step)
        self.tblogger.add_scalar("ce_loss", self.ce_loss.val, self.step)
        self.tblogger.add_scalar("lr", self.curr_lr, self.step)
        self.tblogger.add_scalar("obj_num_avg", self.num_objs.val, self.step)
        # self.tblogger.flush()

    @torch.no_grad()
    def eval_davis(self, split):
        cfg = self.cfg
        self.dist_engine.eval()
        
        seqs_tmp = []
        with open(os.path.join(cfg.DAVIS_ROOT, f'ImageSets/2017/{split}.txt')) as f:
            seqs_tmp.extend(f.readlines())
        sequences = list(map(lambda elem: elem.strip(), seqs_tmp))

        output_root = os.path.join(self.eval_output_root, f"davis2017_{split}")
        for seq_name in sequences:
            self.inference_davis_sequence(seq_name, output_root)

        dataset_eval = DAVISEvaluation(davis_root=cfg.DAVIS_ROOT, task="semi-supervised", gt_set=split)

        metrics_res = dataset_eval.evaluate(os.path.join(output_root, f"masks_{self.step}"))
        
        J, F = metrics_res['J'], metrics_res['F']
        J_global = np.array(J["M"]).mean()
        F_global = np.array(F["M"]).mean()
        J_per_seq = defaultdict(list)
        F_per_seq = defaultdict(list)
        for seq_obj in J['M_per_object'].keys():
            seq = seq_obj.split("_")[0]
            J_per_seq[seq].append(J['M_per_object'][seq_obj])
            F_per_seq[seq].append(F['M_per_object'][seq_obj])
        J_per_seq = {seq: sum(scores) / len(scores) for seq, scores in J_per_seq.items()}
        F_per_seq = {seq: sum(scores) / len(scores) for seq, scores in F_per_seq.items()}

        for seq in J_per_seq.keys():
            print(f"{seq} J-score: {J_per_seq[seq]}; F-score {F_per_seq[seq]}")

        LOGGER.info(f"DAVIS2017 {split} J-score: {J_global:.3f}; F-score {F_global:.3f}; J&F {(J_global+F_global)/2:.3f}")
        self.tblogger.add_scalar(f"davis2017_iou_{split}", J_global, self.step)
        self.tblogger.add_scalar(f"davis2017_f_bound_{split}", F_global, self.step)
        self.tblogger.flush()

        torch.cuda.empty_cache()
        self.dist_engine.train()

    @torch.no_grad()
    def inference_davis_sequence(self, seq_name, output_root):
        cfg = self.cfg

        h0 = 640
        input_image_path = os.path.join(os.path.join(cfg.DAVIS_ROOT, 'Annotations', "480p", seq_name), "00000.png")
        input_image = Image.open(input_image_path)

        w, h = input_image.size
        w = int(w * h0 / h)
        h = h0

        composed_transforms = transforms.Compose([
            # tr.AlignResize((1080, 1920)),
            # tr.AlignResize((80 * 16, 45 * 16)),
            # tr.AlignResize((480, 848)),
            # tr.AlignResize((h, w)),
            tr.Align(),
            tr.ToTensor(),
            ])
        dataset = DAVIS2017Eval(root=cfg.DAVIS_ROOT, seq_name=seq_name, transform=composed_transforms, resolution=cfg.DAVIS_EVAL_RESOLUTION)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        image_root = os.path.join(cfg.DAVIS_ROOT, 'JPEGImages', cfg.DAVIS_EVAL_RESOLUTION, seq_name)

        output_mask_root = os.path.join(output_root, f"masks_{self.step}", seq_name)
        output_video_root = os.path.join(output_root, f"mp4_{self.step}")
        output_video_path = os.path.join(output_video_root, f"{seq_name}_{self.step}.mp4")

        os.makedirs(output_mask_root, exist_ok=True)
        os.makedirs(output_video_root, exist_ok=True)

        for sample in dataloader:

            images = [image.cuda() for image in sample['images']]
            masks = [sample['labels'][0].cuda()]
            obj_nums = sample['obj_num']
            
            masks = self.preprocess_masks(masks)

            with torch.cuda.amp.autocast(enabled=True):
                output_masks = self.dist_engine(images, masks, obj_nums, inference=True)

            for ii, output_mask in enumerate(output_masks):
                output_mask = torch.argmax(output_mask, dim=1, keepdim=True)

                image_name = sample['images_name'][ii][0].split('/')[-1]
                input_image_path = os.path.join(os.path.join(cfg.DAVIS_ROOT, 'JPEGImages', "480p", seq_name), image_name)
                output_mask_path = os.path.join(output_mask_root, f"{image_name.split('.')[0]}.png")

                input_image = Image.open(input_image_path)

                pred_label = output_mask[0, 0].cpu().numpy().astype(np.uint8)
                pred_label = Image.fromarray(pred_label).convert('P')
                pred_label = pred_label.resize(input_image.size)
                pred_label.putpalette(_palette_davis)
                pred_label.save(output_mask_path)

                # pred_label.putpalette(_palette)
                # overlayed_image = _overlay(
                #     np.array(input_image, dtype=np.uint8),
                #     np.array(pred_label, dtype=np.uint8), color_palette)
                # overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

                # if ii == 0:
                #     writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, input_image.size[::-1])
                # writer.write(overlayed_image)
        # writer.release()
        LOGGER.info(f"Created video: {seq_name} - {len(output_masks)} frames")

    @torch.no_grad()
    def inference_ytb_sequence(self, seq_name, output_root):
        cfg = self.cfg

        composed_transforms = transforms.Compose([
            # tr.AlignShort(480),
            # tr.AlignResize((480, 848)),
            tr.Align(),
            tr.ToTensor(),
            ])
        dataset = YoutubeVOS_Eval(root=cfg.YTB_ROOT, seq_name=seq_name, transform=composed_transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        image_root = os.path.join(cfg.YTB_ROOT, 'valid', 'JPEGImages', seq_name)
        output_mask_root = os.path.join(output_root, seq_name)
        os.makedirs(output_mask_root, exist_ok=True)

        for sample in dataloader:

            images = [image.cuda() for image in sample['images']]
            masks = [label.cuda() for label in sample['labels']]
            label_idxs = sample['label_idxs']
            obj_nums = sample['obj_num']
            
            with torch.cuda.amp.autocast(enabled=True):
                output_masks = self.dist_engine(images, masks, obj_nums, inference=True, label_idxs=label_idxs)

            label_idx = 0
            for ii, output_mask in enumerate(output_masks):
                output_mask = torch.argmax(output_mask, dim=1, keepdim=True)

                if ii in label_idxs:
                    output_mask = output_mask * (masks[label_idx].cpu() == 0) + masks[label_idx].cpu()
                    label_idx += 1

                image_name = sample['images_name'][ii][0].split('/')[-1]
                input_image_path = os.path.join(os.path.join(cfg.YTB_ROOT, 'valid', 'JPEGImages', seq_name), image_name)
                output_mask_path = os.path.join(output_mask_root, f"{image_name.split('.')[0]}.png")

                input_image = Image.open(input_image_path)

                pred_label = output_mask[0, 0].cpu().numpy().astype(np.uint8)
                pred_label = Image.fromarray(pred_label).convert('P')
                pred_label = pred_label.resize(input_image.size)
                pred_label.putpalette(_palette_davis)
                pred_label.save(output_mask_path)

        LOGGER.info(f"Created video: {seq_name} - {len(output_masks)} frames")


    @torch.no_grad()
    def eval_ytb(self):
        cfg = self.cfg
        self.dist_engine.eval()
        
        sequences = sorted(os.listdir(os.path.join(cfg.YTB_ROOT, "valid", "JPEGImages")))

        output_root = os.path.join(self.eval_output_root, f"ytb2019_valid", "Annotations")
        for seq_name in tqdm(sequences):
            # if seq_name != "00f88c4f0a":
            #     continue
            self.inference_ytb_sequence(seq_name, output_root)

        torch.cuda.empty_cache()
        self.dist_engine.train()

@torch.no_grad()
    def eval_mose(self):
        cfg = self.cfg
        self.dist_engine.eval()
        
        sequences = sorted(os.listdir(os.path.join(cfg.MOSE_ROOT, "valid", "JPEGImages")))

        output_root = os.path.join(self.eval_output_root, f"mose2023_valid")
        for seq_name in sequences:
            self.inference_mose_sequence(seq_name, output_root)

    @torch.no_grad()
    def inference_mose_sequence(self, seq_name, output_root):
        cfg = self.cfg

        composed_transforms = transforms.Compose([
            tr.AlignShort(480),
            tr.Align(),
            tr.ToTensor(),
            ])
        dataset = MOSE2023_Eval(root=cfg.MOSE_ROOT, seq_name=seq_name, transform=composed_transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        
        output_mask_root = os.path.join(output_root, f"masks_{self.step}", seq_name)
        output_video_root = os.path.join(output_root, f"mp4_{self.step}")
        output_video_path = os.path.join(output_video_root, f"{seq_name}_{self.step}.mp4")

        os.makedirs(output_mask_root, exist_ok=True)
        os.makedirs(output_video_root, exist_ok=True)

        for sample in dataloader:

            images = [image.cuda() for image in sample['images']]
            masks = [sample['labels'][0].cuda()]
            obj_nums = sample['obj_num']

            masks = self.preprocess_masks(masks)

            with torch.cuda.amp.autocast(enabled=True):
                output_masks = self.dist_engine(images, masks, obj_nums, inference=True)

            for ii, output_mask in enumerate(output_masks):
                output_mask = torch.argmax(output_mask, dim=1, keepdim=True)

                image_name = sample['images_name'][ii+1][0].split('/')[-1]
                input_image_path = os.path.join(os.path.join(cfg.MOSE_ROOT, 'valid', 'JPEGImages', seq_name), image_name)
                output_mask_path = os.path.join(output_mask_root, f"{image_name.split('.')[0]}.png")

                input_image = Image.open(input_image_path)

                pred_label = output_mask[0, 0].cpu().numpy().astype(np.uint8)
                pred_label = Image.fromarray(pred_label).convert('P')
                pred_label = pred_label.resize(input_image.size)
                pred_label.putpalette(_palette_davis)
                pred_label.save(output_mask_path)

                pred_label.putpalette(_palette)
                overlayed_image = _overlay(
                    np.array(input_image, dtype=np.uint8),
                    np.array(pred_label, dtype=np.uint8), color_palette)
                overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

                if ii == 0:
                    # writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, input_image.size[::-1])
                    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, input_image.size)
                writer.write(overlayed_image)
        writer.release()

        LOGGER.info(f"Created video: {seq_name} - {len(output_masks)} frames")
