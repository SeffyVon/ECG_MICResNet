from torch.utils.data import Dataset
import torch
import os
import random
random.seed(0)
import numpy as np
from saved_data_io import read_file

class BagSigDataset(Dataset):

    def __init__(self, fDatas, manyhot_encoding_labels, 
        class_idx, stage, n_segments, max_segment_len):
        """
                
        channel_imgs = chn -> array of PImage
        """
        self.fDatas = fDatas
        self.manyhot_encoding_labels = manyhot_encoding_labels
        self.stage = stage
        self.class_idx = class_idx
        self.n_segments = n_segments
        self.max_segment_len = max_segment_len

    def __len__(self):
        return len(self.manyhot_encoding_labels)

    def __getitem__(self, idx):

        fData = self.fDatas[idx]
        manyhot_encoding_label = self.manyhot_encoding_labels[idx]

        if fData.shape[1] < self.max_segment_len:
            fData = np.pad(fData, pad_width=((0,0),(0,self.max_segment_len-fData.shape[1])), mode='constant', constant_values=0)

        j_sigs = 0
        if self.stage  == 'train':
            len_max = fData.shape[1] - self.max_segment_len
            j_sigs = [random.randint(0, len_max) for k in range(self.n_segments)]
        else:
            segment_offset_len = (fData.shape[1] - self.max_segment_len) //(self.n_segments+1)
            j_sigs = [segment_offset_len * k for k in range(self.n_segments)]

        # instances in a bag
        fDatas = np.array([fData[:12,j_sig:j_sig+self.max_segment_len] for j_sig in j_sigs])
        sample =(torch.from_numpy(fDatas).type(torch.FloatTensor),
                 torch.Tensor(manyhot_encoding_label[self.class_idx]))
        return sample
