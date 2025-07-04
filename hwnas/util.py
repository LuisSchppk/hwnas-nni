from collections import OrderedDict
import torch.nn as nn

def replace_conv_bias_with_bn(module: nn.Module, device):
    """
    Recursively traverse `module`, and whenever we find a ConvNd with bias=True,
    replace it by a Sequential block:
        Sequential(
            ConvNd(..., bias=False),
            BatchNormNd(num_features=out_channels)
        )
    The BatchNorm is initialized so that it adds back the original conv bias.
    """

    for name, child in module.named_children():
        # Recurse first
        replace_conv_bias_with_bn(child, device)

        # Check if child is ConvNd with bias
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and child.bias is not None:
            conv_type = type(child)
            in_ch = child.in_channels
            out_ch = child.out_channels
            ksz = child.kernel_size
            stride = child.stride
            padding = child.padding
            dilation = child.dilation
            groups = child.groups

            # Clone old parameters
            old_weight = child.weight.data.clone()
            old_bias = child.bias.data.clone()

            # New conv with bias=False
            new_conv = conv_type(
                in_ch, out_ch,
                kernel_size=ksz,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            )
            new_conv.weight.data.copy_(old_weight)
            new_conv.to(device=device)

            # Create BatchNorm for the appropriate dimension
            if isinstance(child, nn.Conv1d):
                bn = nn.BatchNorm1d(out_ch)
            elif isinstance(child, nn.Conv2d):
                bn = nn.BatchNorm2d(out_ch)
            else:  # Conv3d
                bn = nn.BatchNorm3d(out_ch)

            # Initialize BN to add original bias back
            bn.weight.data.fill_(1.0)
            bn.bias.data.copy_(old_bias)
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)

            bn.to(device=device)
            # Create sequential block conv -> bn
            new_block = nn.Sequential(OrderedDict([("conv1", new_conv), ("bn1", bn)]))
            new_block.to(device=device)

            # Replace original conv with the sequential block
            setattr(module, name, new_block)
            
        elif isinstance(child, nn.Linear) and child.bias is not None:
            in_features = child.in_features
            out_features = child.out_features

            old_weight = child.weight.data.clone()
            old_bias = child.bias.data.clone()

            new_linear = nn.Linear(in_features, out_features, bias=False)
            new_linear.weight.data.copy_(old_weight)

            bn = nn.BatchNorm1d(out_features)
            bn.weight.data.fill_(1.0)
            bn.bias.data.copy_(old_bias)
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)

            new_block = nn.Sequential(OrderedDict([("linear", new_linear), ("bn", bn)]))
            setattr(module, name, new_block)