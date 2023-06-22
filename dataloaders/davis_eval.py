import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from glob import glob

cv2.setNumThreads(0)

class DAVIS2017Eval(Dataset):
    def __init__(
            self, 
            root, 
            seq_name,
            transform=None,
            resolution="480p",
        ):
        self.image_root = os.path.join(root, 'JPEGImages', resolution)
        self.label_root = os.path.join(root, 'Annotations', resolution)
        self.transform = transform

        self.seq_name = seq_name

        self.images = sorted(glob(os.path.join(self.image_root, seq_name, "*.jpg")))
        self.first_label = sorted(glob(os.path.join(self.label_root, seq_name, "*.png")))[0]

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
        first_label = [self.get_label(self.first_label)]
        obj_num = np.max(first_label[0])

        sample = {
            'images': images,
            'labels': first_label,
            'obj_num': obj_num,
            'images_name': [os.path.basename(image) for image in self.images],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
