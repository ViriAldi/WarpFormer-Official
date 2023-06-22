import torch
import torch.nn as nn
import torch.nn.functional as F
from warpformer import WarpFormer


class WarpFormerEngine(nn.Module):
    def __init__(self, cfg, rank):
        super(WarpFormerEngine, self).__init__()
        self.cfg = cfg
        self.rank = rank
        self.model = WarpFormer(cfg).cuda()

        for p in self.model.flow_estimator.parameters():
            p.requires_grad = False
    
    def mask_to_one_hot(self, mask):
        return F.one_hot(mask.long()[:,0,:,:], num_classes=self.cfg.NUM_CLASSES).permute(0, 3, 1, 2).float()

    def forward(self, images, masks, obj_nums, label_idxs=None, inference=False, seq_train=False):
        B, _, H, W = images[0].shape

        self.model.reset()
        logits_preds = []

        if inference:
            p = torch.arange(self.cfg.NUM_CLASSES - 1)
            pinv = torch.argsort(p)

            if label_idxs is None:
                label_idxs = [0]
            label_idx = 0

            if 0 in label_idxs:
                obj_nums = [len(torch.unique(masks[0])) - 1]
                mask_prev = self.mask_to_one_hot(masks[0])
                masks = masks[1:]
                label_idxs = label_idxs[1:]
            else:
                obj_nums = [0]
                mask_prev = self.mask_to_one_hot(torch.zeros_like(masks[0]))
            
            logits_preds.append(mask_prev.clone().cpu())
            
            mask_prev[:,1:] = mask_prev[:,1:][:,p]
            image_prev = images[0]
            images = images[1:]

            self.model.add_reference_img(image_prev)
            self.model.add_reference_mask(mask_prev)
        else:
            p = torch.randperm(self.cfg.NUM_CLASSES - 1)
            pinv = torch.argsort(p)

            image_ref = images[0]
            mask_ref = self.mask_to_one_hot(masks[0])
            mask_ref[:,1:] = mask_ref[:,1:][:,p]
            self.model.add_reference_img(image_ref)
            self.model.add_reference_mask(mask_ref)

            image_prev = images[1]
            mask_prev = self.mask_to_one_hot(masks[1])
            mask_prev[:,1:] = mask_prev[:,1:][:,p]
            self.model.add_reference_img(image_prev)
            self.model.add_reference_mask(mask_prev)

            images = images[2:]
            masks = masks[2:]

        n = len(images)

        for idx in range(n):

            if inference:
                add_memory = (idx % 5 == 4) or (inference and idx in label_idxs)
            else:
                add_memory = idx % 2 == 1

            logits_pred = self.model(images[idx], image_prev, mask_prev, add_to_memory=add_memory)

            logits_pred[:,1:] = logits_pred[:,1:][:,pinv]
            for ii in range(B):
                logits_pred[ii, obj_nums[ii]+1:] = -10000
            if inference:
                logits_preds.append(logits_pred.clone().cpu())
            else:
                logits_preds.append(logits_pred.clone())
            logits_pred[:,1:] = logits_pred[:,1:][:,p]

            if seq_train or inference:
                mask_prev = torch.argmax(logits_pred, dim=1, keepdim=True)
                mask_prev = self.mask_to_one_hot(mask_prev)
            else:
                mask_prev = self.mask_to_one_hot(masks[idx])
                mask_prev[:,1:] = mask_prev[:,1:][:,p]
            image_prev = images[idx]

            if add_memory:
                if (inference and idx in label_idxs):
                    mask_prev = torch.argmax(mask_prev, dim=1, keepdim=True)
                    mask_prev = mask_prev * (masks[label_idx] == 0) + masks[label_idx]
                    mask_prev = self.mask_to_one_hot(mask_prev[:,0])

                    obj_nums[0] += len(torch.unique(masks[label_idx])) - 1
                    label_idx += 1
                
                self.model.add_reference_mask(mask_prev)

        self.model.reset()

        return logits_preds
