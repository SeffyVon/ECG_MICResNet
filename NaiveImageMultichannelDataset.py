from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import numbers


class NaiveImageMultichannelDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, channel_imgs, manyhot_encoding_labels, class_idx, stage):
        """
                
        channel_imgs = chn -> array of PImage
        """
        self.manyhot_encoding_labels = manyhot_encoding_labels
        self.channel_imgs = channel_imgs
        self.transform =  {
            'train':  transforms.Compose([
                                 transforms.RandomCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ]),  
            'test': transforms.Compose([
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])}[stage]
        self.class_idx = class_idx

    def __len__(self):
        return len(self.manyhot_encoding_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = [self.channel_imgs[idx][chn] for chn in range(4)]
        manyhot_encoding_label = self.manyhot_encoding_labels[idx]
        sample =(torch.cat([self.transform(image) for image in images],0), 
                 torch.Tensor(manyhot_encoding_label[self.class_idx]))

        return sample

