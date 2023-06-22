import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import json

cv2.setNumThreads(0)

class YoutubeVOS_Eval(Dataset):
    def __init__(
            self, 
            root, 
            seq_name,
            transform=None,
        ):
        root = os.path.join(root, 'valid')
        self.image_root = os.path.join(root, 'JPEGImages')
        self.label_root = os.path.join(root, 'Annotations')

        self.transform = transform

        self.seq_name = seq_name

        # self.seq_list_file = os.path.join(root, 'meta.json')
        # with open(self.seq_list_file) as f:
            # self.ann_f = json.load(f)['videos'][seq_name]
        # images = set()
        # labels = set()
        # for obj_n, obj_data in self.ann_f['objects'].keys():
        #     images.update([os.path.join(self.image_root, seq_name, f"{frame}.jpg") for frame in obj_data["frames"]])
        #     labels.update([os.path.join(self.label_root, seq_name, f"{frame}.png") for frame in obj_data["frames"]])
        # images = sorted(images)
        # labels = sorted(labels)

        self.images = []
        self.labels = []
        self.label_idxs = []
        for idx, path in enumerate(sorted(glob(os.path.join(self.image_root, seq_name, "*.jpg")))):
            frame = os.path.splitext(os.path.basename(path))[0]
            self.images.append(path)
            label_path = os.path.join(self.label_root, seq_name, f"{frame}.png")
            if os.path.exists(label_path):
                self.labels.append(label_path)   
                self.label_idxs.append(idx)     

    def __len__(self):
        return 1
    
    def get_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)

        return image
    
    def get_label(self, path):
        label = Image.open(path)
        label = np.array(label, dtype=np.uint8)

        return label
    
    def __getitem__(self, idx):
        images = [self.get_image(image) for image in self.images]
        labels = [self.get_label(label) for label in self.labels]
        obj_num = sum(len(np.unique(label)) - 1 for label in labels)

        sample = {
            'images': images,
            'labels': labels,
            'label_idxs': self.label_idxs,
            'obj_num': obj_num,
            'images_name': [os.path.basename(image) for image in self.images],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
