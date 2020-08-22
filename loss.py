    
import torch
import torch.nn as nn
 
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss

class DSCLoss(nn.Module):
    def __init__(self):
        super(DSCLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        numerator = 2 * ((1-input_flat)*input_flat*target_flat).sum(1) + smooth 
        denominator = ((1-input_flat)*input_flat + target_flat).sum(1) + smooth
        loss = 1-numerator/denominator
 
        return loss.sum()

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    def __init__(self, weights=None):
        super(MulticlassDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, input, target):
 
        C = target.shape[1]
 
        if self.weights is None:
            self.weights = torch.ones(C) #uniform weights for all classes

        # set-typed DL from paper
        # smooth = 1
        # loss = 1- (2*(input*target).sum(1) + smooth) / ((input**2).sum(1) + (target**2).sum(1) + smooth)
        totalLoss = 0
        dl = DiceLoss()
        for i in range(C):
            dscloss = dl(input[:,i], target[:,i])
            dscloss *= self.weights[i]
            totalLoss += dscloss

        return totalLoss

class MulticlassDSCLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    def __init__(self, weights=None):
        super(MulticlassDSCLoss, self).__init__()
        self.weights = weights

    def forward(self, input, target):
 
        C = target.shape[1]
 
        if self.weights is None:
            self.weights = torch.ones(C) #uniform weights for all classes
 
        dsc = DSCLoss()
        totalLoss = 0
        for i in range(C):
            dscloss = dsc(input[:,i], target[:,i])
            dscloss *= self.weights[i]
            totalLoss += dscloss
        return totalLoss

        # smooth = 1
        # numerator = 2 * ((1-input)*input*target).sum(1) + smooth 
        # denominator = ((1-input)*input + target).sum(1) + smooth
        # loss = 1-numerator/denominator
        # loss = loss.sum()
        # return loss

def weighted_binary_cross_entropy2(sigmoid_x, y, weighted_matrix, weight=None, reduction=None):
    """
    Aha this is correct!
    sigmoid_x = nn.Sigmoid()(x)
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1]
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (y.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(y.size(), sigmoid_x.size()))
   
    #print("y.size(), sigmoid_x.size()", y.size(), sigmoid_x.size())
    sigmoid_x = torch.clamp(sigmoid_x,min=1e-7,max=1-1e-7) 
    loss = - torch.matmul(y*sigmoid_x.log() + (1-y)*(1-sigmoid_x).log(), weighted_matrix)
    
    if weight is not None:
        loss = loss * weight
        
    if reduction is None:
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return None
    
class WeightedBCELoss(nn.Module):
    def __init__(self, weights, PosWeightIsDynamic= False, WeightIsDynamic= False, 
                 reduction='mean'):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weights', weights)
        self.reduction = reduction
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input, target):
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)


        return weighted_binary_cross_entropy2(input, target,
                                             weighted_matrix=self.weights,
                                             reduction=self.reduction)