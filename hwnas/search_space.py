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

class VGG8ModelSpaceCIFAR10(ModelSpace):
    def __init__(self):
        NUM_CLASSES = 10
        super().__init__()

        out_size = nni.choice("out_size_conv1", [16, 32, 64])
        mult = nni.choice("channel_multiplier", [1, 1.5, 2])
        kernel_size_conv1 = nni.choice('kernel_size_conv1', [3, 5, 7])
        kernel_size_conv2 = nni.choice('kernel_size_conv2', [3, 5, 7])
        kernel_size_conv3 = nni.choice('kernel_size_conv3', [3, 5])
        kernel_size_conv4 = nni.choice('kernel_size_conv4', [3, 5])
        kernel_size_conv7 = nni.choice('kernel_size_conv7', [1, 3])
        # kernel_size_pool = nni.choice('pool_size', [2,4])

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
    

class VGG8ModelSpaceCIFAR10OneShot(ModelSpace):
    """
    VGG-8-style model-space for CIFAR-10, explicitly listing all Conv2d variants
    instead of using MutableConv2d.
    """
    def __init__(self):
        super().__init__()
        NUM_CLASSES = 10

        # a mutable for number of output channels
        out_size = nni.choice("out_size_conv1", [64, 128])
        # self.out_size = 64
        self.add_mutable(out_size)
        self.out_size = ensure_frozen(out_size)
        self.out_size_choice = out_size

        xbar_size_choice = nni.choice('xbar_size', [128, 256, 512])
        self.add_mutable(xbar_size_choice)
        self.xbar_size = ensure_frozen(xbar_size_choice)
        self.xbar_size_choice = xbar_size_choice

        # conv1: kernel_size choices [3,5,7]
        self.conv1 = LayerChoice([
            nn.Conv2d(3, self.out_size, kernel_size=3, padding=1),
            nn.Conv2d(3, self.out_size, kernel_size=5, padding=2),
            nn.Conv2d(3, self.out_size, kernel_size=7, padding=3)
        ], label="conv1")

        # conv2: DepthwiseSeparable or standard Conv variants matching conv1 kernel options
        self.conv2 = LayerChoice([
            DepthwiseSeparableConv(self.out_size, self.out_size * 2),
            nn.Conv2d(self.out_size, self.out_size * 2, kernel_size=3, padding=1),
            nn.Conv2d(self.out_size, self.out_size * 2, kernel_size=5, padding=2),
            nn.Conv2d(self.out_size, self.out_size * 2, kernel_size=7, padding=3)
        ], label="conv2")

        # conv3: DepthwiseSeparable or standard Conv variants [3,5]
        self.conv3 = LayerChoice([
            DepthwiseSeparableConv(self.out_size * 2, self.out_size * 4),
            nn.Conv2d(self.out_size * 2, self.out_size * 4, kernel_size=3, padding=1),
            nn.Conv2d(self.out_size * 2, self.out_size * 4, kernel_size=5, padding=2)
        ], label="conv3")

        # conv4: mutable conv or identity
        self.conv4 = LayerChoice([
            nn.Conv2d(self.out_size * 4, self.out_size * 4, kernel_size=3, padding=1),
            nn.Conv2d(self.out_size * 4, self.out_size * 4, kernel_size=5, padding=2),
            nn.Identity()
        ], label="conv4")

        # conv5: two choices of kernel 3 or 5
        self.conv5 = LayerChoice([
            nn.Conv2d(self.out_size * 4, self.out_size * 8, kernel_size=3, padding=1),
            nn.Conv2d(self.out_size * 4, self.out_size * 8, kernel_size=5, padding=2)
        ], label="conv5")

        # conv6: conv or identity
        self.conv6 = LayerChoice([
            nn.Conv2d(self.out_size * 8, self.out_size * 8, kernel_size=3, padding=1),
            nn.Identity()
        ], label="conv6")

        # conv7: always 3x3 no padding
        self.conv7 = nn.Conv2d(self.out_size * 8, self.out_size * 16, kernel_size=3, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        # final classifier
        final_spatial_size = 1
        fc_in = self.out_size * 16 * final_spatial_size
        self.fc1 = nn.Linear(fc_in, NUM_CLASSES, bias=False)

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
    

# adapted from tutortial
class TutorialModelSpace(ModelSpace):
    def __init__(self):
        super().__init__()

        NUM_CLASSES = 10
        self.conv1 = nn.Conv2d(3, 32, 3, 1)

        kernel_size_conv2 = nni.choice('kernel_size_conv2', [3, 5])
        self.conv2 = LayerChoice([
            MutableConv2d(32, 64, kernel_size=kernel_size_conv2, padding=(kernel_size_conv2 // 2)),
            # DepthwiseSeparableConv(32, 64)
        ], label='conv2')

        kernel_size_conv3 = nni.choice('kernel_size_conv3', [3, 5])
        self.conv3 = LayerChoice([
            MutableConv2d(64, 128, kernel_size=kernel_size_conv3, padding=(kernel_size_conv3 // 2)),
            # DepthwiseSeparableConv(64, 128)
        ], label='conv3')

        self.pool = LayerChoice([
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        ], label='pool_choice')

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        # self.dropout1 = MutableDropout(nni.choice('dropout', [0.25])) 
        # self.dropout2 = MutableDropout(nni.choice('dropout', [0.75]))

        feature = 10
        self.fc1 = nn.Linear(128, feature)  # dynamically fixed below
        self.fc2 = Linear(feature, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x