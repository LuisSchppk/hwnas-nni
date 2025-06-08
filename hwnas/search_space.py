import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import (
    ModelSpace, LayerChoice, InputChoice, Repeat, 
    MutableDropout, MutableLinear, MutableConv2d
)

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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class VGG8ModelSpaceCIFAR10(ModelSpace):
    """
    VGG-8-style model-space for CIFAR-10 (3x32x32 → 10 classes), with:
      - LayerChoice over conv2/conv3 (standard vs. depthwise-separable vs. MutableConv2d),
      - InputChoice between a “skip” (upsampled) path and the main path,
      - MutableDropout and MutableLinear,
      - MutableConv2d (as one of the conv candidates).
    """
    def __init__(self):
        NUM_CLASSES = 10
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        kernel_size_conv2 = nni.choice('kernel_size_conv2', [3, 5])
        self.conv2 = LayerChoice([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            DepthwiseSeparableConv(64, 128),
            MutableConv2d(64, 128, kernel_size=kernel_size_conv2, padding=(kernel_size_conv2 // 2))
        ], label='conv2_choice')

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32×32 → 16×16

        kernel_size_conv3 = nni.choice('kernel_size_conv3', [3, 5])
        self.conv3 = LayerChoice([
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            DepthwiseSeparableConv(128, 256),
            MutableConv2d(128, 256, kernel_size=kernel_size_conv3, padding=(kernel_size_conv3 // 2))
        ], label='conv3_choice')

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 → 8x8

        # self.skip_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.skip_conv = nn.Conv2d(64, 256, kernel_size=1, bias=False)

        # self.input_choice = InputChoice(n_candidates=1, label="input_choice1")
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)  # 8x8 → 4x4

        self.dropout1 = MutableDropout(nni.choice('dropout1', [0.25, 0.5, 0.75]))
        self.dropout2 = MutableDropout(nni.choice('dropout2', [0.25, 0.5, 0.75]))

        fc_in_features = 256 * 4 * 4
        hidden_dim = nni.choice('fc_hidden_dim', [256, 512])
        self.fc1 = MutableLinear(fc_in_features, hidden_dim)
        self.fc2 = MutableLinear(hidden_dim, NUM_CLASSES)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # [B, 64, 32, 32]
        x1_p = self.pool1(x1)       # [B, 64, 16, 16]

        x2 = self.conv2(x1)         # could be one of {Conv2d, DW‐SepConv, MutableConv2d}
        x2 = F.relu(x2)             # must be [B, 128, 32, 32] before pooling
        x2_p = self.pool1(x2)       # [B, 128, 16, 16]

        x3 = self.conv3(x2_p)       # before relu should be [B, 256, 16, 16]
        x3 = F.relu(x3)             # [B, 256, 16, 16]
        x3_p = self.pool2(x3)       # [B, 256, 8, 8]

        # skip = self.skip_pool(x1_p)   # where skip_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #                               # [B, 64, 8, 8]
        # skip = self.skip_conv(skip)   # [B, 256, 8, 8]
        # skip = F.relu(skip)


        # x_choice = self.input_choice([x3_p, skip])  # [B, 256, 8, 8]

        x4 = self.pool4(x3_p)        # [B, 256, 4, 4]
        x4 = torch.flatten(x4, 1)        # [B, 256*4*4]

        x4 = self.dropout1(x4)
        x4 = F.relu(self.fc1(x4))        # [B, hidden_dim]
        x4 = self.dropout2(x4)
        logits = self.fc2(x4)            # [B, num_classes]
        return logits
    

# adapted from tutortial
class TutorialModelSpace(ModelSpace):
    def __init__(self):
        super().__init__()

        NUM_CLASSES = 10
        self.conv1 = nn.Conv2d(3, 32, 3, 1)

        kernel_size_conv2 = nni.choice('kernel_size_conv2', [3, 5])
        self.conv2 = LayerChoice([
            MutableConv2d(32, 64, kernel_size=kernel_size_conv2, padding=(kernel_size_conv2 // 2)),
            DepthwiseSeparableConv(32, 64)
        ], label='conv2')

        kernel_size_conv3 = nni.choice('kernel_size_conv3', [3, 5])
        self.conv3 = LayerChoice([
            MutableConv2d(64, 128, kernel_size=kernel_size_conv3, padding=(kernel_size_conv3 // 2)),
            DepthwiseSeparableConv(64, 128)
        ], label='conv3')

        self.pool = LayerChoice([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ], label='pool_choice')

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = MutableDropout(nni.choice('dropout', [0.25, 0.5, 0.75])) 
        self.dropout2 = MutableDropout(nni.choice('dropout', [0.25, 0.5, 0.75]))

        feature = nni.choice('feature', [32, 64, 128])
        self.fc1 = MutableLinear(128, feature)  # dynamically fixed below
        self.fc2 = MutableLinear(feature, NUM_CLASSES)

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