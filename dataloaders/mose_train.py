import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch
import random

cv2.setNumThreads(0)

class MOSE2023_Train(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        seq_len=1,
        repeat_time=1,
        rand_gap=4,
        ):
        self.image_root = os.path.join(root, 'train', 'JPEGImages')
        self.label_root = os.path.join(root, 'train', 'Annotations')
        self.transform = transform

        self.seq_names = os.listdir(self.image_root)

        self.seq_len = seq_len
        self.repeat_time = repeat_time
        self.rand_gap = rand_gap

        self.imglistdict = {}
        for seq_name in self.seq_names:
            images = list(np.sort(os.listdir(os.path.join(self.image_root, seq_name))))
            images = list(map(lambda path: os.path.join(seq_name, path), images))
            labels = list(np.sort(os.listdir(os.path.join(self.label_root, seq_name))))
            labels = list(map(lambda path: os.path.join(seq_name, path), labels))

            assert len(images) == len(labels)
            if len(images) > 7:
                self.imglistdict[seq_name] = (images, labels)
        self.seq_names = list(self.imglistdict.keys())

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
    
    def get_ref_index(self, lablist, prev_index, obj_nums, min_fg_pixels=100):
        if prev_index == 0:
            return 0, self.get_label(lablist[0])

        bad_indices = set()
        idx = 0
        while idx < 100:
            idx += 1
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
        return self.sample_sequence(idx)
