from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import numbers
from typing import Tuple
import random
random.seed(0)
from torchvision.transforms.transforms import _get_image_size
from torchvision.transforms import functional as F
import numpy as np

sub_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


from PIL import Image
from saved_data_io import read_file
class IdvImageSigDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, output_directory, filenames, manyhot_encoding_labels, class_idx, stage='train'):
        """
                
        channel_imgs = chn -> array of PImage
        """
        self.output_directory = output_directory
        self.manyhot_encoding_labels = manyhot_encoding_labels
        self.filenames = filenames
        self.stage = stage
        self.class_idx = class_idx

    def __len__(self):
        return len(self.manyhot_encoding_labels)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        images = read_file(self.output_directory+ '/cwt/' + filename + '.pkl')
        fData = read_file(self.output_directory+ '/sig/' + filename + '.npy')

        manyhot_encoding_label = self.manyhot_encoding_labels[idx]
        j = 0 # left
        _, w = _get_image_size(images[0])
        if self.stage in ['train', 'result']:
            j = random.randint(0, w - 224)
        else:
            # central crop
            j = max((w - 224 ) //2, 0)

        j_sig = max(0, min(int(j/224 * 3000), fData.shape[1]-3000))
        if fData[:,j_sig:j_sig+3000].shape[1] != 3000:
            #print('error:', idx, j_sig, fData.shape[1], fData[:,j_sig:j_sig+3000].shape[1])
            fData = np.pad(fData, pad_width=((0,0),(0,3000-fData.shape[1])), mode='constant', constant_values=0)
            #print(fData.shape)
            #print('error[after]:', fData.shape[1], fData[:,j_sig:j_sig+3000].shape[1])
        # if fData[:,j_sig:j_sig+3000].shape[0] != 12:
        #     print('error:', idx, filename, fData.shape[0])
        #     pass
        sample =(torch.cat([sub_transform(F.crop(img, 0, j, 224, 224)) for img in images], 0), 
                 torch.from_numpy(fData[:12,j_sig:j_sig+3000]).type(torch.FloatTensor),
                 torch.Tensor(manyhot_encoding_label[self.class_idx]))
        return sample

class IdvSigDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, output_directory, filenames, manyhot_encoding_labels, class_idx, stage='train'):
        """
                
        channel_imgs = chn -> array of PImage
        """
        self.output_directory = output_directory
        self.manyhot_encoding_labels = manyhot_encoding_labels
        self.filenames = filenames
        self.stage = stage
        self.class_idx = class_idx

    def __len__(self):
        return len(self.manyhot_encoding_labels)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        fData = read_file(self.output_directory+ '/sig/' + filename + '.npy')

        manyhot_encoding_label = self.manyhot_encoding_labels[idx]
        
        j_sig = 0
        if self.stage in ['train', 'result']:
            j_sig = random.randint(0, max(fData.shape[1] - 3000, 0))
        else:
            j_sig = max((fData.shape[1] - 3000 ) //2, 0)

        if fData[:,j_sig:j_sig+3000].shape[1] != 3000:
            #print('error:', idx, j_sig, fData.shape[1], fData[:,j_sig:j_sig+3000].shape[1])
            fData = np.pad(fData, pad_width=((0,0),(0,3000-fData.shape[1])), mode='constant', constant_values=0)
            #print(fData.shape)
            #if self.stage == 'test':
             #   print('error[after]:', fData.shape[1], fData[:,j_sig:j_sig+3000].shape[1])
        # if fData[:,j_sig:j_sig+3000].shape[0] != 12:
        #     print('error:', idx, filename, fData.shape[0])
        #     pass
        sample =(torch.from_numpy(fData[:12,j_sig:j_sig+3000]).type(torch.FloatTensor),
                 torch.Tensor(manyhot_encoding_label[self.class_idx]))
        return sample

