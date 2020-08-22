"""
1D ResNet adapted from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
"""
import torch
from torch import nn

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

class ResNet333Block(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3conv/batchnorm/activation
    """
    
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
             conv_bn(self.in_channels, self.in_channels, conv=self.conv, kernel_size=3),
             activation_func(self.activation),
             conv_bn(self.in_channels, self.in_channels, conv=self.conv, kernel_size=3, stride=self.downsampling),
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
        if n == 1:
            self.blocks = nn.Sequential(
                block(in_channels, out_channels, downsampling=1, *args, **kwargs) 
        )
        else:
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
                 block, n, activation='relu', *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=1, activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for k, (in_channels, out_channels) in enumerate(self.in_out_block_sizes)]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResNetEncoderFixed(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels, blocks_sizes,
                 block, n, activation='relu', *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes[1:-1], blocks_sizes[2:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[1], n=1, activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels, 
                          out_channels, n=n, activation=activation, 
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
    

class ECGResNet(nn.Module):
    
    """
    Combining 12 lead ecg on a network (actually not 50)
    """
    
    def __init__(self, in_channels, n_classes, n=3, blocks_sizes=[64, 128, 128], *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, block=ResNet753Block, blocks_sizes=blocks_sizes, n=n, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].out_channels, n_classes)
       
    def forward(self, x):
        x = self.encoder(x) 
        x = self.decoder(x)
        return x

class ECGFeatureResNet(nn.Module):
    def __init__(self, in_channels, n_features, n_classes, 
            blocks_sizes=[64, 128, 256], verbose=False):
        super().__init__()
        self.resnet_encoder = nn.Sequential(
            ResNetEncoder(in_channels, block=ResNet753Block, 
            blocks_sizes=blocks_sizes, n=3),
            nn.AdaptiveAvgPool1d((1,)),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features,n_features),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            )
        self.decoder = nn.Linear(blocks_sizes[-1] + n_features, n_classes)
        self.verbose = verbose


    def forward(self, x1, x2):
        x1 = self.resnet_encoder(x1)
        if self.verbose:
            print("x1.shape", x1.shape)
        x1 = x1.view((x1.shape[0], x1.shape[1]*x1.shape[2]))
        if self.verbose:
            print("x1.shape", x1.shape)
        x2 = self.feature_encoder(x2)
        if self.verbose:
            print("x2.shape", x2.shape)
        x = torch.cat([x1, x2], 1)
        if self.verbose:
            print("x.shape", x.shape)
        x = self.decoder(x)
        return x

import torch.nn.functional as F
class ECGBagResNet(nn.Module):
    
    """
    Combining 12 lead ecg on a network (actually not 50)
    """
    
    def __init__(self, in_channels, n_classes, n_segments, n=3,
            blocks_sizes=[64, 128, 128], verbose=False):
        super().__init__()
        encoder_dim = blocks_sizes[-1]
        D = encoder_dim//2
        K = encoder_dim//4
        # D = encoder_dim
        # K = encoder_dim
        self.encoder = nn.Sequential(
            ResNetEncoder(in_channels, block=ResNet753Block, 
            blocks_sizes=blocks_sizes, n=3),
            nn.AdaptiveAvgPool1d((1,)),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(encoder_dim, D),
            nn.Tanh(),
            nn.Linear(D, K)
        )
        # ll.bias = nn.Parameter(init_bias)
        self.decoder = nn.Linear(4096, n_classes)
        #self.decoder = nn.Linear(16384, n_classes) #

        self.verbose = verbose
        self.n_segments = n_segments

    def forward(self, xs):
        H = [self.encoder(xs[:,i,:,:]).view((xs.shape[0],-1,1)) for i in range(self.n_segments)]
        if self.verbose:
            print("0 H[0].shape", H[0].shape)
        H = torch.cat(H, dim=2) # batch x channels x n_segments 
        if self.verbose:
            print("cat H.shape", H.shape)
        H = torch.transpose(H, 1, 2) # batch x n_segments x channels 
        if self.verbose:
            print("transpose H.shape", H.shape)


        A = self.attention(H) # batch x n_segments x channels_out
        if self.verbose:
            print("attention A.shape", A.shape)
        A = torch.transpose(A, 1, 2) # batch  x channels_out x n_segments
        if self.verbose:
            print("transpose A.shape", A.shape)
        A = F.softmax(A, dim=1) # batch  x channels_out x n_segments
        if self.verbose:
            print("softmax A.shape", A.shape)


        M = torch.bmm(A, H)
        if self.verbose:
            print("bmm M.shape", M.shape)
        M = M.view(M.size(0), -1)
        if self.verbose:
            print("view M.shape", M.shape)
        y_prob = self.decoder(M)
        return y_prob    

