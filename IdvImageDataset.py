from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import numbers
from typing import Tuple
import random
from torchvision.transforms.transforms import _get_image_size
from torchvision.transforms import functional as F


class MultiRandomCrop(transforms.RandomCrop):

    def __init__(self, size, sub_transform, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        
        super().__init__(size)
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.sub_transform = sub_transform

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop. based on 

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, imgs):
        """
        Args:
            imgs (list of PIL Image): list of Images to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        img = imgs[0]
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        
        return torch.cat([self.sub_transform(F.crop(img, i, j, h, w)) for img in imgs], 0)

class MultiCenterCrop(transforms.CenterCrop):

    def __init__(self, size, sub_transform):
        super().__init__(size)
        self.sub_transform = sub_transform

    def __call__(self,imgs):
        """
        Args:
            imgs (list of PIL Image): list of Images to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return torch.cat([sub_transform(F.center_crop(img, self.size)) for img in imgs], 0)


sub_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
img_transforms = {
            'train':  MultiRandomCrop(224, sub_transform),  
            'test': MultiCenterCrop(224, sub_transform),
            'result': MultiRandomCrop(224, sub_transform),
    }

from PIL import Image
from saved_data_io import read_file
class IdvImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, output_directory, filenames, manyhot_encoding_labels, class_idx, stage='train'):
        """
                
        channel_imgs = chn -> array of PImage
        """
        self.output_directory = output_directory
        self.manyhot_encoding_labels = manyhot_encoding_labels
        self.filenames = filenames
        self.transform = img_transforms[stage]
        self.class_idx = class_idx

    def __len__(self):
        return len(self.manyhot_encoding_labels)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        images = read_file(self.output_directory+ '/cwt/' + filename + '.pkl')
        manyhot_encoding_label = self.manyhot_encoding_labels[idx]
        sample =(self.transform(images), 
                 torch.Tensor(manyhot_encoding_label[self.class_idx]))

        return sample


