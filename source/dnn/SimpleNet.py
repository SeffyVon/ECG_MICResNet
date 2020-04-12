import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, n_input, n_h1, n_h2, n_h3, verbose=False):
        super(SimpleNet, self).__init__()
        self.fc1        = nn.Linear(n_input, n_h1)
        self.fc2        = nn.Linear(n_h1, n_h2)
        self.fc3        = nn.Linear(120, n_h3)
        self.fc4        = nn.Linear(n_h3, 9)
        self.dropout = nn.Dropout(p=0.5)
        self.verbose=verbose

    def forward(self, x):
        if self.verbose:
            print('0: ', x.shape)
        x = self.fc1( x )     
        x = self.dropout(x)
        if self.verbose:
            print('1: ', x.shape)
        x = F.relu(   x )     
        x = self.fc2( x )    
        if self.verbose:
            print('2: ', x.shape)
        x = F.relu(   x )      
        if self.verbose:
            print('3: ', x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc3( x )  
        if self.verbose:
            print('4: ', x.shape)
        x = F.relu(   x )   
        x = self.fc4(x)
        if self.verbose:
            print('5: ', x.shape)
        return x
