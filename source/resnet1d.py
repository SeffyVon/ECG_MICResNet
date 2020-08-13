"""
1D ResNet adapted from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
"""
import torch
from torch import nn
from torchvision import models

from functools import partial
class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2,) # dynamic add padding based on the kernel_size
        #print(self.kernel_size, self.padding)
        
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual 
        x = self.activate(x)
        return x
    

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, downsampling=1, kernel_size=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.downsampling, self.conv = downsampling, partial(Conv1dAuto, kernel_size=kernel_size, bias=False)   
        self.shortcut = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm1d(self.out_channels))
        

def conv_bn(in_channels, out_channels, conv, kernel_size, *args, **kwargs):
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size=kernel_size, *args, **kwargs), 
        nn.BatchNorm1d(out_channels)
        )

        
class ResNet753Block(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3conv/batchnorm/activation
    """
    
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
             conv_bn(self.in_channels, self.in_channels, conv=self.conv, kernel_size=7),
             activation_func(self.activation),
             conv_bn(self.in_channels, self.in_channels, conv=self.conv, kernel_size=5, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.in_channels, out_channels, conv=self.conv, kernel_size=3),
        )

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , in_channels, *args, **kwargs, downsampling=downsampling),
            *[block(in_channels, in_channels, downsampling=1, *args, **kwargs) for _ in range(1, n-1)],
            block(in_channels, out_channels, downsampling=1, *args, **kwargs) 
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels, blocks_sizes,
                 block, activation='relu', *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, blocks_sizes[0], kernel_size=9, bias=False),
            nn.BatchNorm1d(blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3),
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=1, activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels, 
                          out_channels, n=3, activation=activation, 
                          block=block, *args, **kwargs) 
              for k, (in_channels, out_channels) in enumerate(self.in_out_block_sizes)]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d((1,))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x
    

class ECGResNet50(nn.Module):
    
    """
    Combining 12 lead ecg on a network
    """
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, block=ResNet753Block, blocks_sizes=[64, 128, 128], *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].out_channels, n_classes)
       
    def forward(self, x):
        #x = self.conv1(x)
        x = self.encoder(x) 
        x = self.decoder(x)
        return x
    
