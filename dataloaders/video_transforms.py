import random
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter

cv2.setNumThreads(0)


class AlignResize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample["labels"][0].shape[:2]
        if self.output_size == (h, w):
            return sample
        else:
            new_h, new_w = self.output_size

        sample["images"] = [cv2.resize(x, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR) for x in sample["images"]]
        sample["labels"] = [cv2.resize(x, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST) for x in sample["labels"]]

        return sample


class AlignShort(object):
    def __init__(self, short_side):
        self.short_side = short_side

    def __call__(self, sample):
        h, w = sample["labels"][0].shape[:2]

        if h < w:
            new_h = self.short_side
            new_w = int((w / h) * new_h)
        else:
            new_w = self.short_side
            new_h = int((h / w) * new_w)
        new_w = new_w - new_w % 16
        new_h = new_h - new_h % 16

        sample["images"] = [cv2.resize(x, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR) for x in sample["images"]]
        sample["labels"] = [cv2.resize(x, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST) for x in sample["labels"]]

        return sample


class Align(object):
    def __call__(self, sample):
        h, w = sample["labels"][0].shape[:2]
        dh, dw = h % 16, w % 16
        if (dh, dw) == (0, 0):
            return sample
        else:
            new_h, new_w = (h-dh, w-dw)

        sample["images"] = [cv2.resize(x, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR) for x in sample["images"]]
        sample["labels"] = [cv2.resize(x, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST) for x in sample["labels"]]

        return sample

class ToTensor(object):
    def __call__(self, sample):

        sample["images"] = [self.img2imagenet(x) for x in sample["images"]]
        sample["images"] = [x.transpose((2, 0, 1)) for x in sample["images"]]
        sample["images"] = [torch.from_numpy(x).to(torch.float32) for x in sample["images"]]

        sample["labels"] = [x[:, :, np.newaxis] for x in sample["labels"]]
        sample["labels"] = [x.transpose((2, 0, 1)) for x in sample["labels"]]
        sample["labels"] = [torch.from_numpy(x).int() for x in sample["labels"]]

        return sample

    def img2imagenet(self, img):
        img = img / 255.
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)
    
        return img


class RandomScale(object):
    def __init__(self, min_scale=1., max_scale=1.3, short_edge=None):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.short_edge = short_edge

    def __call__(self, sample):
        sc = np.random.uniform(self.min_scale, self.max_scale)

        if self.short_edge is not None:
            image = sample['images'][0]
            h, w = image.shape[:2]
            if h > w:
                sc *= float(self.short_edge) / w
            else:
                sc *= float(self.short_edge) / h

        sample['images'] = [cv2.resize(x, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR) for x in sample['images']]
        sample['labels'] = [cv2.resize(x, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST) for x in sample['labels']]

        return sample


class BalancedRandomCrop(object):
    def __init__(self,
                 output_size,
                 max_step=5,
                 min_obj_pixel_num=100):
        self.output_size = output_size
        self.max_step = max_step
        self.min_obj_pixel_num = min_obj_pixel_num

    def __call__(self, sample):

        image = sample['images'][0]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = min(h, new_h)
        new_w = min(w, new_w)

        is_contain_obj = False
        step = 0
        while (not is_contain_obj) and (step < self.max_step):
            step += 1
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            after_crop = []
            contains = []
            for elem in (sample['labels']):
                tmp = elem[top:top + new_h, left:left + new_w]
                contains.append(np.unique(tmp))
                after_crop.append(tmp)

            all_obj = list(np.sort(contains[0]))
            if all_obj[-1] == 0:
                continue
            if all_obj[0] == 0:
                all_obj = all_obj[1:]
            new_all_obj = []
            for obj_id in all_obj:
                after_crop_pixels = np.sum(after_crop[0] == obj_id)
                if after_crop_pixels > self.min_obj_pixel_num:
                    new_all_obj.append(obj_id)

            is_contain_obj = len(new_all_obj) != 0
            all_obj = [0] + new_all_obj

        post_process = []
        for elem in after_crop:
            new_elem = elem * 0
            for idx in range(1, len(all_obj)):
                obj_id = all_obj[idx]
                mask = elem == obj_id
                new_elem += (mask * idx).astype(np.uint8)
            post_process.append(new_elem.astype(np.uint8))

        sample['labels'] = post_process
        sample['images'] = [x[top:top + new_h, left:left + new_w] for x in sample['images']]
        sample['obj_num'] = len(all_obj) - 1

        return sample


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.p = prob

    def __call__(self, sample):

        if random.random() < self.p:
            sample['images'] = [cv2.flip(x, flipCode=1) for x in sample['images']]
            sample['labels'] = [cv2.flip(x, flipCode=1) for x in sample['labels']]

        return sample


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class RandomGaussianBlur(object):
    def __init__(self, prob=0.3, sigma=[0.1, 2.0]):
        self.aug = torchvision.transforms.RandomApply([GaussianBlur(sigma)], p=prob)

    def __call__(self, sample):
        sample["images"] = [self.apply_augmentation(x) for x in sample['images']]
        return sample

    def apply_augmentation(self, x):
        x = Image.fromarray(np.uint8(x))
        x = self.aug(x)
        x = np.array(x, dtype=np.float32)
        return x

class RandomGrayScale(RandomGaussianBlur):
    def __init__(self, prob=0.2):
        self.aug = torchvision.transforms.RandomGrayscale(p=prob)

class RandomColorJitter(RandomGaussianBlur):
    def __init__(self, prob=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.aug = torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)], p=prob)


class Padding(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample['images'][0].shape[:2]
        if self.output_size == (h, w):
            return sample
        else:
            new_h, new_w = self.output_size

        def sep_pad(x):
            x0 = np.random.randint(0, x + 1)
            x1 = x - x0
            return x0, x1

        tp, bp = sep_pad(new_h - h)
        lp, rp = sep_pad(new_w - w)

        sample['images'] = [cv2.copyMakeBorder(x, tp, bp, lp, rp, cv2.BORDER_CONSTANT, value=(124, 116, 104)) for x in sample['images']]
        sample['labels'] = [cv2.copyMakeBorder(x, tp, bp, lp, rp, cv2.BORDER_CONSTANT, value=(0)) for x in sample['labels']]

        return sample
