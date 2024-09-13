import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, Sampler
import os
import glob
import random
from collections import defaultdict
from PIL import Image

class TextClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_path_list = glob.glob(self.data_dir + '/*/*') 
        self.classes = os.listdir(self.data_dir)  
        self.num_classes = len(self.classes)
        self.classes_dict = {v: k for k, v in enumerate(self.classes)}
        for folder in self.classes:
            folder_path = os.path.join(self.data_dir, folder)
            num_samples = len(glob.glob(folder_path + '/*'))
            print(f"Folder '{folder}' contains {num_samples} samples.")

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_name = img_path.split('/')[-2]  # Extract the folder name (class)

        # Create one-hot encoded label
        label = np.zeros(self.num_classes, dtype=np.float32)
        
        # Assign labels based on folder names
        if label_name == 'signature':
            label[self.classes_dict[label_name]] = 1
            label[self.classes_dict['hw']] = 1
        else:
            label[self.classes_dict[label_name]] = 1  

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)

    def __len__(self):
        return len(self.img_path_list)




class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(32, 320), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, image):
        """Return image with added border."""
        img = image.copy()
        img = np.array(img)
        shape = img.shape[:2]  # current shape [height, width]        

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (self.new_shape[1], self.new_shape[0])

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
        # cv2.namedWindow("test_after_letterbox", cv2.WINDOW_NORMAL)
        # cv2.imshow('test_after_letterbox', img)
        # cv2.waitKey()
        # print(img.shape)
        img = Image.fromarray(img)
        return img


def preprocess(images, transform: LetterBox, device, half=False):
    """ Preprocess the images (using LetterBox transform) before pass into yolov8 model"""
    not_tensor = not isinstance(images, torch.Tensor)
    if not_tensor:
        images = np.stack([transform(image=x) for x in images])
        images = images[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        images = np.ascontiguousarray(images)  # contiguous
        images = torch.from_numpy(images)

    images = images.to(device)
    images = images.half() if half else images.float()
    if not_tensor:
        images /= 255  # 0 - 255 to 0.0 - 1.0
    return images

