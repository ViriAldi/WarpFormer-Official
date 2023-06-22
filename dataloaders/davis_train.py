import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch
import random

cv2.setNumThreads(0)

class DAVIS2017Train(Dataset):
    def __init__(
        self,
        root,
        splits=["train", "val"],
        transform=None,
        seq_len=1,
        repeat_time=5,
        resolution="480p",
        rand_gap=4,
        merge_prob=0.5,
        ):
        self.image_root = os.path.join(root, 'JPEGImages', resolution)
        self.label_root = os.path.join(root, 'Annotations', resolution)
        self.transform = transform

        seq_names = []
        for split in splits:
            with open(os.path.join(root, "ImageSets", "2017", split + ".txt")) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seq_names.extend(seqs_tmp)
        self.seq_names = seq_names

        self.seq_len = seq_len
        self.repeat_time = repeat_time
        self.rand_gap = rand_gap
        self.merge_prob = merge_prob

        self.imglistdict = {}
        for seq_name in self.seq_names:
            images = list(np.sort(os.listdir(os.path.join(self.image_root, seq_name))))
            images = list(map(lambda path: os.path.join(seq_name, path), images))
            labels = list(np.sort(os.listdir(os.path.join(self.label_root, seq_name))))
            labels = list(map(lambda path: os.path.join(seq_name, path), labels))

            assert len(images) == len(labels)
            self.imglistdict[seq_name] = (images, labels)

    def __len__(self):
        return int(len(self.seq_names) * self.repeat_time)
    
    def get_image(self, path):
        image = cv2.imread(os.path.join(self.image_root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)

        return image
    
    def get_label(self, path):
        label = Image.open(os.path.join(self.label_root, path))
        label = np.array(label, dtype=np.uint8)

        return label
    
    def get_ref_index(self, lablist, prev_index, obj_nums, min_fg_pixels=200):
        if prev_index == 0:
            return 0, self.get_label(lablist[0])

        bad_indices = set()
        while True:
            ref_index = random.randint(0, prev_index)
            if ref_index in bad_indices:
                continue

            ref_label = self.get_label(lablist[ref_index])
            if len(np.unique(ref_label)) - 1 == obj_nums:
                if np.count_nonzero(ref_label) > min_fg_pixels:
                    break
    
            bad_indices.add(ref_index)
        return ref_index, ref_label
    
    def get_prev_index(self, lablist, total_gap):
        search_range = len(lablist) - total_gap - 1
        return random.randint(0, search_range)
    
    def get_curr_gaps(self, lablist):
        curr_gaps = []
        total_gap = 0
        rand_gap = max(min(self.rand_gap, (len(lablist)-1) // self.seq_len), 1)
        for _ in range(self.seq_len):
            gap = random.randint(1, rand_gap)
            total_gap += gap
            curr_gaps.append(gap)
        return curr_gaps, total_gap
    
    def sample_sequence(self, idx):
        idx = idx % len(self.seq_names)
        seq_name = self.seq_names[idx]
        imagelist, lablist = self.imglistdict[seq_name]
        obj_nums = np.max(self.get_label(lablist[0]))

        # generate random gaps
        curr_gaps, total_gap = self.get_curr_gaps(lablist)
        prev_index = self.get_prev_index(lablist, total_gap)

        ref_index, ref_label = self.get_ref_index(lablist, prev_index, obj_nums)

        idxs = [ref_index, prev_index]
        cur_idx = prev_index
        for gap in curr_gaps:
            cur_idx += gap
            idxs.append(cur_idx)

        images = [self.get_image(imagelist[ii]) for ii in idxs]
        labels = [ref_label] + [self.get_label(lablist[ii]) for ii in idxs[1:]]

        sample = {
            'images': images,
            'labels': labels,
            'obj_num': obj_nums,
            'images_name': [os.path.basename(imagelist[idx]) for idx in idxs],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        idx = np.random.randint(len(self.seq_names))
        sample1 = self.sample_sequence(idx)

        if random.random() < self.merge_prob:
            rand_idx = np.random.randint(len(self.seq_names))
            while (rand_idx == (idx % len(self.seq_names))):
                rand_idx = np.random.randint(len(self.seq_names))

            sample2 = self.sample_sequence(rand_idx)

            sample = self.merge_sample(sample1, sample2)
        else:
            sample = sample1

        return sample
    
    def merge_sample(self, sample1, sample2, max_obj_n=10, min_obj_pixels=100):
        obj_idx = torch.arange(0, max_obj_n * 2 + 1).view(max_obj_n * 2 + 1, 1, 1)
        selected_idx = None
        selected_obj = None

        all_img = []
        all_mask = []
        for idx in range(len(sample1["labels"])):
            s1_img = sample1["images"][idx]
            s2_img = sample2["images"][idx]
            
            s1_label = sample1["labels"][idx]
            s2_label = sample2["labels"][idx]
            s2_fg = (s2_label > 0).float()
            s2_bg = 1 - s2_fg

            merged_img = None
            if s1_img is not None and s2_img is not None:
                merged_img = s1_img * s2_bg + s2_img * s2_fg

            merged_mask = s1_label * s2_bg.long() + (s2_label + max_obj_n) * s2_fg.long()
            merged_mask = (merged_mask == obj_idx).float()
            if idx == 0:
                after_merge_pixels = merged_mask.sum(dim=(1, 2), keepdim=True)
                selected_idx = after_merge_pixels > min_obj_pixels
                selected_idx[0] = True
                obj_num = selected_idx.sum().int().item() - 1
                selected_idx = selected_idx.expand(-1, s1_label.shape[1], s1_label.shape[2])

                if obj_num > max_obj_n:
                    selected_obj = list(range(1, obj_num + 1))
                    random.shuffle(selected_obj)
                    selected_obj = [0] + selected_obj[:max_obj_n]

            merged_mask = merged_mask[selected_idx].view(obj_num + 1, s1_label.shape[1], s1_label.shape[2])
            if obj_num > max_obj_n:
                merged_mask = merged_mask[selected_obj]
            merged_mask = torch.argmax(merged_mask, dim=0, keepdim=True).long()

            if merged_img is not None:  
                all_img.append(merged_img)
            all_mask.append(merged_mask)

        sample = {
            'images': all_img,
            'labels': all_mask,
            'obj_num': obj_num,
            'images_name': sample1['images_name'],
        }
        return sample
