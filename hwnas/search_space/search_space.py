import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import (
    ModelSpace, LayerChoice, InputChoice, Repeat, 
    MutableDropout, MutableLinear, MutableConv2d, MutableMaxPool2d
)

from nni.mutable.symbol import SymbolicExpression 

from nni.mutable import ensure_frozen

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = MutableConv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_ch)
        self.conv2 =  MutableConv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv2(self.conv1(x))


class VGG8ModelSpaceVGG8(ModelSpace):
    """
    Model Space defining a space of VGG8 like models. Based on VGG8 Architecture used in MNSIM-2.0
    """
    def __init__(self):
        NUM_CLASSES = 10
        super().__init__()

        out_size = nni.choice("out_size_conv1", [16, 32, 64])
        mult = nni.choice("channel_multiplier", [1, 1.5, 2])
        kernel_size_conv1 = nni.choice('kernel_size_conv1', [3, 5, 7])
        kernel_size_conv2 = nni.choice('kernel_size_conv2', [3, 5, 7])
        kernel_size_conv3 = nni.choice('kernel_size_conv3', [3, 5])
        kernel_size_conv4 = nni.choice('kernel_size_conv4', [3, 5])
        kernel_size_conv7 = 3
        # kernel_size_conv7 = nni.choice('kernel_size_conv7', [1, 3])

        self.conv1 = MutableConv2d(3, out_size, kernel_size=kernel_size_conv1, padding=(kernel_size_conv1 // 2) )

        self.conv2 = LayerChoice([
            DepthwiseSeparableConv(out_size, SymbolicExpression.to_int(out_size * mult) ),
            MutableConv2d( out_size, SymbolicExpression.to_int(out_size * mult), kernel_size=kernel_size_conv2, padding=(kernel_size_conv2 // 2) )
        ], label='conv2_choice')

        self.conv3 = LayerChoice([
            DepthwiseSeparableConv(
                SymbolicExpression.to_int(out_size * mult),
                SymbolicExpression.to_int(out_size * mult * mult)
            ),
            MutableConv2d(
                SymbolicExpression.to_int(out_size * mult),
                SymbolicExpression.to_int(out_size * mult * mult),
                kernel_size=kernel_size_conv3,
                padding=(kernel_size_conv3 // 2)
            )
        ], label='conv3_choice')

        self.conv4 = LayerChoice([
            MutableConv2d(
                SymbolicExpression.to_int(out_size * mult * mult),
                SymbolicExpression.to_int(out_size * mult * mult),
                kernel_size=kernel_size_conv4,
                padding=(kernel_size_conv4 // 2)
            ),
            nn.Identity()
        ], label='conv4_choice')

        self.conv5 = MutableConv2d(
            SymbolicExpression.to_int(out_size * mult * mult),
            SymbolicExpression.to_int(out_size * mult * mult * mult),
            kernel_size=3,
            padding=1
        )

        self.conv6 = LayerChoice([
            MutableConv2d(
                SymbolicExpression.to_int(out_size * mult * mult * mult),
                SymbolicExpression.to_int(out_size * mult * mult * mult),
                kernel_size=3,
                padding=1
            ),
            nn.Identity()
        ], label='conv6_choice')

        self.conv7 = MutableConv2d(
            SymbolicExpression.to_int(out_size * mult * mult * mult),
            SymbolicExpression.to_int(out_size * mult * mult * mult * mult),
            kernel_size=kernel_size_conv7,
            padding=0
        )

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool = LayerChoice([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ], label='pool_choice')

        self.relu = nn.ReLU(inplace=True)

        final_spatial_size = 1
        fc_in_features = SymbolicExpression.to_int(
                out_size * mult * mult * mult * mult * final_spatial_size * final_spatial_size
        )

        self.fc1 = MutableLinear(fc_in_features, NUM_CLASSES, bias=False)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))

        if not isinstance(self.conv4, nn.Identity):
            x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.relu(self.conv5(x))
        if not isinstance(self.conv6, nn.Identity):
            x = self.relu(self.conv6(x))
        x = self.pool(x)

        x = self.relu(self.conv7(x))
        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1) 
        x = self.fc1(x)
        return x
    

class VGG8ModelSpaceVGG8XBar(ModelSpace):
    """
    Model Space defining a space of VGG8 like models. Based on VGG8 Architecture used in MNSIM-2.0. Additionally this model space includes a parameter for XBar size.
    """
    def __init__(self):
        NUM_CLASSES = 10
        super().__init__()

        out_size = nni.choice("out_size_conv1", [16, 32, 64])
        mult = nni.choice("channel_multiplier", [1, 1.5, 2])
        kernel_size_conv1 = nni.choice('kernel_size_conv1', [3, 5, 7])
        kernel_size_conv2 = nni.choice('kernel_size_conv2', [3, 5, 7])
        kernel_size_conv3 = nni.choice('kernel_size_conv3', [3, 5])
        kernel_size_conv4 = nni.choice('kernel_size_conv4', [3, 5])
        kernel_size_conv7 = 3
        # kernel_size_conv7 = nni.choice('kernel_size_conv7', [1, 3])

        xbar_size_choice = nni.choice('xbar_size', [128, 256, 512])
        self.add_mutable(xbar_size_choice)
        self.xbar_size = ensure_frozen(xbar_size_choice)
        self.xbar_size_choice = xbar_size_choice

        self.conv1 = MutableConv2d(3, out_size, kernel_size=kernel_size_conv1, padding=(kernel_size_conv1 // 2) )

        self.conv2 = LayerChoice([
            DepthwiseSeparableConv(out_size, SymbolicExpression.to_int(out_size * mult) ),
            MutableConv2d( out_size, SymbolicExpression.to_int(out_size * mult), kernel_size=kernel_size_conv2, padding=(kernel_size_conv2 // 2) )
        ], label='conv2_choice')

        self.conv3 = LayerChoice([
            DepthwiseSeparableConv(
                SymbolicExpression.to_int(out_size * mult),
                SymbolicExpression.to_int(out_size * mult * mult)
            ),
            MutableConv2d(
                SymbolicExpression.to_int(out_size * mult),
                SymbolicExpression.to_int(out_size * mult * mult),
                kernel_size=kernel_size_conv3,
                padding=(kernel_size_conv3 // 2)
            )
        ], label='conv3_choice')

        self.conv4 = LayerChoice([
            MutableConv2d(
                SymbolicExpression.to_int(out_size * mult * mult),
                SymbolicExpression.to_int(out_size * mult * mult),
                kernel_size=kernel_size_conv4,
                padding=(kernel_size_conv4 // 2)
            ),
            nn.Identity()
        ], label='conv4_choice')

        self.conv5 = MutableConv2d(
            SymbolicExpression.to_int(out_size * mult * mult),
            SymbolicExpression.to_int(out_size * mult * mult * mult),
            kernel_size=3,
            padding=1
        )

        self.conv6 = LayerChoice([
            MutableConv2d(
                SymbolicExpression.to_int(out_size * mult * mult * mult),
                SymbolicExpression.to_int(out_size * mult * mult * mult),
                kernel_size=3,
                padding=1
            ),
            nn.Identity()
        ], label='conv6_choice')

        self.conv7 = MutableConv2d(
            SymbolicExpression.to_int(out_size * mult * mult * mult),
            SymbolicExpression.to_int(out_size * mult * mult * mult * mult),
            kernel_size=kernel_size_conv7,
            padding=0
        )

        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pool = LayerChoice([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ], label='pool_choice')

        self.relu = nn.ReLU(inplace=True)

        final_spatial_size = 1
        fc_in_features = SymbolicExpression.to_int(
                out_size * mult * mult * mult * mult * final_spatial_size * final_spatial_size
        )

        self.fc1 = MutableLinear(fc_in_features, NUM_CLASSES, bias=False)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))

        if not isinstance(self.conv4, nn.Identity):
            x = self.relu(self.conv4(x))
        x = self.pool(x)

        x = self.relu(self.conv5(x))
        if not isinstance(self.conv6, nn.Identity):
            x = self.relu(self.conv6(x))
        x = self.pool(x)

        x = self.relu(self.conv7(x))
        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1) 
        x = self.fc1(x)
        return x