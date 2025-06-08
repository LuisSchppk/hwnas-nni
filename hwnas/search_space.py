import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import (
    ModelSpace, LayerChoice, InputChoice, Repeat, 
    MutableDropout, MutableLinear, MutableConv2d
)

# 
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv2(self.conv1(x))
    

class ElementMultiplicationLayer(nn.Module):
    """
        MNSIM-2.0 parses element_mul layers into this class. 
        To ensure model is translated correctly to MNSIM-2.0 all element multiplication has to be realised using this class.
    """
    
    def __init__(self):
        super(ElementMultiplicationLayer, self).__init__()
    def forward(self, x):
        x[0]=x[0].view(x[0].shape[0],x[0].shape[1],1,1)
        return x[0]*x[1]

# adapted from tutortial
class TutorialModelSpace(ModelSpace):
    def __init__(self):
        super().__init__()

        NUM_CLASSES = 100
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv2 = LayerChoice([
            # nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ], label='conv2')

        # nni.choice is used to select a dropout rate.
        # The result can be used as parameters of `MutableXXX`.
        self.dropout1 = MutableDropout(nni.choice('dropout', [0.25]))  # choose dropout rate from 0.25, 0.5 and 0.75
        self.dropout2 = nn.Dropout(0.5)
        final_size = 1 # restrict to one due to MNSIM 2.0
        self.pool = nn.AdaptiveAvgPool2d((final_size, final_size))
        flattend_final_size = 64 * (final_size ** 2)

        feature = nni.choice('feature', [256])
        self.fc1 = MutableLinear(flattend_final_size, feature)
        self.fc2 = MutableLinear(feature, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.pool(x)            
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output