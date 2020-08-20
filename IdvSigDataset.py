from torch.utils.data import Dataset
import torch
import os
import random
random.seed(0)
import numpy as np

from saved_data_io import read_file


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
            fData = np.pad(fData, pad_width=((0,0),(0,3000-fData.shape[1])), mode='constant', constant_values=0)
   
        sample =(torch.from_numpy(fData[:12,j_sig:j_sig+3000]).type(torch.FloatTensor),
                 torch.Tensor(manyhot_encoding_label[self.class_idx]))
        return sample

class IdvSigFeatureDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, output_directory, filenames, features, manyhot_encoding_labels, class_idx, stage='train'):
        """
                
        channel_imgs = chn -> array of PImage
        """
        self.output_directory = output_directory
        self.manyhot_encoding_labels = manyhot_encoding_labels
        self.filenames = filenames
        self.stage = stage
        self.class_idx = class_idx
        self.features = features

    def __len__(self):
        return len(self.manyhot_encoding_labels)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        fData = read_file(self.output_directory+ '/sig/' + filename + '.npy')
        feature = self.features[idx] #read_file(self.output_directory+ '/features/' + filename + '.npy')

        manyhot_encoding_label = self.manyhot_encoding_labels[idx]
        
        j_sig = 0
        if self.stage == 'train':
            j_sig = random.randint(0, max(fData.shape[1] - 3000, 0))
        else:
            j_sig = max((fData.shape[1] - 3000 ) //2, 0)

        if fData[:,j_sig:j_sig+3000].shape[1] != 3000:
            fData = np.pad(fData, pad_width=((0,0),(0,3000-fData.shape[1])), mode='constant', constant_values=0)
   
        sample =(torch.from_numpy(fData[:12,j_sig:j_sig+3000]).type(torch.FloatTensor),
                 torch.from_numpy(feature).type(torch.FloatTensor),
                 torch.Tensor(manyhot_encoding_label[self.class_idx]))
        return sample

