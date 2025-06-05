import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import (
    ModelSpace, LayerChoice, InputChoice, Repeat, 
    MutableDropout, MutableLinear, MutableConv2d
)

def compute_final_size(k, kernel_size, padding, pool_kernel, pool_stride, num_layers):
    """Helper function to compute final feature map size"""
    size = k
    for _ in range(num_layers):
        size = size 
        size = (size - pool_kernel) // pool_stride + 1
    return size

class CIFAR10NetFCGRModelSpace(ModelSpace):
    """
    !!!Generated using generative AI based on base model!!!

    NNI Model Space based on CIFAR10NetFCGR
    Makes key hyperparameters and architectural choices searchable
    """
    
    def __init__(self, num_classes, k):
        super().__init__()
        
        # Searchable hyperparameters
        self.kernel_size = nni.choice('kernel_size', [3, 5, 7])
        self.pool_kernel = nni.choice('pool_kernel', [2, 3])
        self.pool_stride = nni.choice('pool_stride', [2])
        self.num_layers = nni.choice('num_layers', [2, 3, 4])
        
        # Searchable channel dimensions
        self.conv1_channels = nni.choice('conv1_channels', [16, 32, 48])
        self.conv2_channels = nni.choice('conv2_channels', [32, 64, 96])
        self.conv3_channels = nni.choice('conv3_channels', [64, 128, 192])
        self.hidden_size = nni.choice('hidden_size', [128, 256, 512])
        
        # Searchable activation function
        self.activation = nni.choice('activation', ['relu', 'leaky_relu', 'elu', 'swish'])
        
        # Searchable dropout rate
        self.dropout_rate = nni.choice('dropout_rate', [0.0, 0.2, 0.3, 0.5])
        
        # Calculate conservative final size for worst case (largest kernel, most layers)
        max_final_size = compute_final_size(k, 7, 1, 2, 2, 4)
        
        # Feature extraction layers with searchable components
        
        # First conv block
        self.conv1 = MutableConv2d(
            1, 
            self.conv1_channels,
            kernel_size=self.kernel_size,
            padding=self._get_padding(self.kernel_size)
        )
        
        # Second conv block
        self.conv2 = MutableConv2d(
            self.conv1_channels,
            self.conv2_channels, 
            kernel_size=self.kernel_size,
            padding=self._get_padding(self.kernel_size)
        )
        
        # Third conv block (optional based on num_layers)
        self.conv3 = LayerChoice([
            nn.Identity(),  # Skip when num_layers < 3
            MutableConv2d(
                self.conv2_channels,
                self.conv3_channels,
                kernel_size=self.kernel_size,
                padding=self._get_padding(self.kernel_size)
            )
        ], label='conv3_choice')
        
        # Fourth conv block (optional based on num_layers)
        self.conv4 = LayerChoice([
            nn.Identity(),  # Skip when num_layers < 4
            MutableConv2d(
                self.conv3_channels,
                self.conv3_channels,  # Keep same channels
                kernel_size=self.kernel_size,
                padding=self._get_padding(self.kernel_size)
            )
        ], label='conv4_choice')
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        self.pool3 = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        self.pool4 = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        
        # Batch normalization choice
        self.use_batch_norm = nni.choice('use_batch_norm', [True, False])
        self.bn1 = nn.BatchNorm2d(self.conv1_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2_channels)
        self.bn3 = nn.BatchNorm2d(self.conv3_channels)
        self.bn4 = nn.BatchNorm2d(self.conv3_channels)
        
        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier with searchable components
        self.flatten = nn.Flatten()
        
        # Determine the input size for first linear layer based on final conv channels
        # We'll use the maximum possible channels for safety
        max_conv_channels = max([192, 192])  # max of conv3_channels possibilities
        
        self.fc1 = MutableLinear(
            max_conv_channels * 4 * 4,  # 4x4 from adaptive pooling
            self.hidden_size
        )
        
        self.dropout = MutableDropout(self.dropout_rate)
        
        self.fc2 = MutableLinear(self.hidden_size, num_classes)
        
        # Store choices for forward pass logic
        self._store_choices()
    
    def _get_padding(self, kernel_size):
        """Calculate appropriate padding to maintain spatial dimensions"""
        if isinstance(kernel_size, int):
            return kernel_size // 2
        return 1  # fallback
    
    def _store_choices(self):
        """Store choices for use in forward pass"""
        # These will be resolved when the model is instantiated by NNI
        pass
    
    def _get_activation(self):
        """Get the chosen activation function"""
        if self.activation == 'relu':
            return F.relu
        elif self.activation == 'leaky_relu':
            return F.leaky_relu
        elif self.activation == 'elu':
            return F.elu
        elif self.activation == 'swish':
            return lambda x: x * torch.sigmoid(x)
        else:
            return F.relu  # fallback
    
    def forward(self, x):
        activation_fn = self._get_activation()
        
        # First conv block (always present)
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = activation_fn(x)
        x = self.pool1(x)
        
        # Second conv block (always present)
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = activation_fn(x)
        x = self.pool2(x)
        
        # Third conv block (conditional)
        x = self.conv3(x)
        if not isinstance(self.conv3, nn.Identity) and self.use_batch_norm:
            x = self.bn3(x)
        if not isinstance(self.conv3, nn.Identity):
            x = activation_fn(x)
            x = self.pool3(x)
        
        # Fourth conv block (conditional)
        x = self.conv4(x)
        if not isinstance(self.conv4, nn.Identity) and self.use_batch_norm:
            x = self.bn4(x)
        if not isinstance(self.conv4, nn.Identity):
            x = activation_fn(x)
            x = self.pool4(x)
        
        # Adaptive pooling to standardize size
        x = self.adaptive_pool(x)
        
        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleCIFAR10ModelSpace(ModelSpace):
    """
    Simplified version that more closely follows the original structure
    but with key searchable parameters
    """
    
    def __init__(self):
        super().__init__()
        NUM_CLASSES = 100
        
        kernel_size = nni.choice('kernel_size', [3, 5, 7])
        padding = kernel_size
        pool_kernel = nni.choice('pool_kernel', [2, 3])
        pool_stride = 2
        
        ch1 = nni.choice('ch1', [16, 32, 48])
        ch2 = nni.choice('ch2', [32, 64, 96]) 
        ch3 = nni.choice('ch3', [64, 128, 192])
        hidden_size = nni.choice('hidden_size', [128, 256, 512])
        
        final_size = nni.choice('final_size', [1, 2, 4])
        
        self.features = nn.Sequential(
            MutableConv2d(1, ch1, kernel_size=kernel_size, padding=padding),
            LayerChoice([
                nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                nn.ELU(inplace=True)
            ], label='activation1'),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            
            MutableConv2d(ch1, ch2, kernel_size=kernel_size, padding=padding),
            LayerChoice([
                nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True), 
                nn.ELU(inplace=True)
            ], label='activation2'),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            
            MutableConv2d(ch2, ch3, kernel_size=kernel_size, padding=padding),
            LayerChoice([
                nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                nn.ELU(inplace=True)
            ], label='activation3'),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),

            nn.AdaptiveAvgPool2d((final_size, final_size))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            MutableLinear(ch3 * final_size * final_size, hidden_size),
            LayerChoice([
                nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
                nn.ELU(inplace=True)
            ], label='activation_fc'),
            MutableDropout(nni.choice('dropout_rate', [0.0, 0.2, 0.3, 0.5])),
            MutableLinear(hidden_size, NUM_CLASSES),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# !!! AI GENERATED !!! 
class CompatibleModelSpace(ModelSpace):
    """
    A small NAS search space using only blocks that our parser can handle:
      • Conv2d  (with optional depthwise‐separable)
      • ReLU
      • MaxPool2d
      • AdaptiveAvgPool2d((1,1)) as ADA pooling
      • Flatten (view)
      • Dropout
      • Linear (fc)
      • (Optionally) Sigmoid or SiLU (Swish), or element_sum / element_multiply if added manually
    """
    def __init__(self):
        super().__init__()
        NUM_CLASSES = 100
        # First conv block: always a regular 3×3, stride=1, padding=1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Pooling choice: either MaxPool2d(2,2) or AdaptiveAvgPool2d((1,1))
        self.pool1 = LayerChoice([
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        ], label='pool1')

        # Second conv block: choice between regular Conv2d or DepthwiseSeparableConv
        self.conv2 = LayerChoice([
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1)
        ], label='conv2_choice')
        self.relu2 = nn.ReLU()

        # Second pooling (always MaxPool2d)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten (view)
        self.flatten = nn.Flatten()

        # Choose hidden dimension for fc1
        hidden = nni.choice('hidden_size', [64, 128, 256])

        # After two poolings (assuming input 3×64×64 → conv1 → relu → pool1 reduces to either 32×32 or 1×1):
        # If pool1 == MaxPool2d(2,2), size is (32×32); if pool1 == AdaptiveAvgPool2d((1,1)), size is (1×1).
        # To keep dimensions consistent, we force pool1 to be MaxPool2d in most trials; when pool1 is ADA, pool2
        # will also reduce any spatial dim to (1,1). Here we assume input=3×64×64 and pool1=MaxPool2d, so:
        # 64×64 → pool1(2) → 32×32 → conv2 → relu2 → pool2(2) → 16×16 → flatten → 64*16*16=16384 features.
        # For simplicity, we set fc1's in_features to the maximum possible (when pool1=MaxPool2d):
        self.fc1 = MutableLinear(64 * 16 * 16, hidden)

        # Dropout before final classification
        self.dropout = MutableDropout(nni.choice('drop_rate', [0.25, 0.5]))

        # Final output layer
        self.fc2 = MutableLinear(hidden, NUM_CLASSES)

    def forward(self, x):
        # Conv1 → ReLU
        x = self.relu1(self.conv1(x))

        # Pool1: either MaxPool2d or ADA AvgPool
        x = self.pool1(x)

        # If pool1 was ADA (→ shape: [B, 32, 1, 1]), skip directly to flatten; otherwise continue:
        if x.ndim == 4 and (x.shape[2] != 1 or x.shape[3] != 1):
            # Conv2 → ReLU → Pool2
            x = self.relu2(self.conv2(x))
            x = self.pool2(x)
        else:
            # If pool1 already collapsed spatial dims to (1,1), apply conv2 with stride=1 padding=0, then no further pooling.
            x = self.relu2(self.conv2(x))
            # Optionally, you could still pool2 with ADA to (1,1), but we skip it here.
        
        # Flatten (view)
        x = self.flatten(x)

        # Fully connected → Dropout → Final FC
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# adapted from tutortial
class TutorialModelSpace(ModelSpace):
    def __init__(self):
        super().__init__()

        NUM_CLASSES = 100
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # LayerChoice is used to select a layer between Conv2d and DwConv.
        self.conv2 = LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ], label='conv2')
        # nni.choice is used to select a dropout rate.
        # The result can be used as parameters of `MutableXXX`.
        self.dropout1 = MutableDropout(nni.choice('dropout', [0.25, 0.5, 0.75]))  # choose dropout rate from 0.25, 0.5 and 0.75
        self.dropout2 = nn.Dropout(0.5)
        final_size = 1 # restrict to one due to MNSIM 2.0
        self.pool = nn.AdaptiveAvgPool2d((final_size, final_size))
        flattend_final_size = 64 * (final_size ** 2)

        feature = nni.choice('feature', [64, 128, 256])
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
        output = F.log_softmax(x, dim=1)
        return output