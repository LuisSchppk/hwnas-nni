from collections import OrderedDict
import operator
import re
import torch
import torch.nn as nn
from torch.fx import symbolic_trace


import torch
import torch.nn as nn



def get_next_bn_index(module, scheduled_changes):
    """Compute next available bn index by scanning both existing modules and pending changes."""
    max_idx = 0

    # Check existing bn names
    for name in module._modules:
        match = re.match(r'bn(\d+)', name)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)

    # Check already scheduled bn names in changes
    for _, _, scheduled_bn_name, _ in scheduled_changes:
        match = re.match(r'bn(\d+)', scheduled_bn_name)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)

    return max_idx + 1
    
def replace_conv_bias_with_bn_old(module: nn.Module):
    """
    Recursively traverse `module`, and whenever we find a ConvNd with bias=True,
    replace it by:
        Sequential(
            ConvNd(..., bias=False),
            BatchNormNd(num_features=out_channels)
        )
    The new BatchNorm’s weight (gamma) is initialized to 1, its bias (beta) is set to
    the old conv.bias, and running_mean=0, running_var=1 so that at inference time
    BN simply adds the original bias back.
    """

    changes = list()
    for name, child in module.named_children():
        # Recurse first so that nested blocks get fixed top‐to‐bottom
        replace_conv_bias_with_bn(child)


        # Check whether `child` is a Conv layer with bias=True
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and child.bias is not None:
            # Gather original parameters
            conv_type = type(child)
            in_ch   = child.in_channels
            out_ch  = child.out_channels
            ksz     = child.kernel_size
            stride  = child.stride
            padding = child.padding
            dilation= child.dilation
            groups  = child.groups
            # original bias and weight
            old_weight = child.weight.data.clone()
            old_bias   = child.bias.data.clone()

            # Build a new conv with bias=False, same other args
            new_conv = conv_type(
                in_ch, out_ch,
                kernel_size=ksz,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False  # <— drop bias here
            )
            # Copy over weight
            new_conv.weight.data.copy_(old_weight)

            # Build a BatchNorm on `out_ch` channels
            if isinstance(child, nn.Conv1d):
                bn = nn.BatchNorm1d(out_ch)
            elif isinstance(child, nn.Conv2d):
                bn = nn.BatchNorm2d(out_ch)
            else:  # Conv3d
                bn = nn.BatchNorm3d(out_ch)

            # Initialize BN so that:
            #   γ (bn.weight) = 1
            #   β (bn.bias)   = old conv bias
            #   running_mean = 0, running_var = 1  (identity normalization)
            bn.weight.data.fill_(1.0)
            bn.bias.data.copy_(old_bias)
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)

            orig_conv_name = name            # e.g. "conv1"
            if orig_conv_name.startswith("conv"):
                suffix  = orig_conv_name[len("conv"):]  # e.g. "1"
                bn_name = "bn" + suffix                 # e.g. "bn1"
            else:
                # fallback to bn<next_idx>
                next_idx = get_next_bn_index(module, changes)
                bn_name = f"bn{next_idx}"
            

            # Replace old with new
            # setattr(module, orig_conv_name, new_conv)
            # setattr(module, bn_name, bn)
            
            changes.append((name, new_conv, bn_name, bn))

            # Assign back into parent module
            # setattr(module, name, new_block)

           
    if isinstance(module, nn.Sequential):
        new_order = OrderedDict()
        for k, v in module._modules.items():
            found = False
            for conv_name, conv_layer, bn_name, bn_layer in changes:
                if k == conv_name:
                    new_order[conv_name] = conv_layer
                    new_order[bn_name] = bn_layer
                    found = True
                    break
            if not found:
                new_order[k] = v
        module._modules = new_order
    else:
        for conv_name, conv_layer, bn_name, bn_layer in changes:
            setattr(module, conv_name, conv_layer)
            setattr(module, bn_name, bn_layer)


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

def trace_model(model, num_classes=None, input_shape = None,):
    if input_shape is not None:
        dummy_input = torch.randn(*input_shape)
        traced = symbolic_trace(model, concrete_args={"x" : dummy_input})
    else:
        traced = symbolic_trace(model)
    
    layer_config_list = []
    node_to_idx = {} 
    layer_idx = 0

    def record_layer(config_dict):
        nonlocal layer_idx
        layer_config_list.append(config_dict)
        node_to_idx[node.name] = layer_idx
        layer_idx += 1
        
    for node in traced.graph.nodes:
        if node.op == 'placeholder':
            node_to_idx[node.name] = layer_idx
        elif node.op == 'call_method':
            if node.target == torch.Tensor.view:
                record_layer({'type': 'view'})
            else:
                print("no match for", node, node.target, "in call_method")
        elif node.op == 'call_function':
            if node.target == operator.getitem:

                base, idx = node.args
                if base.op == 'placeholder' and isinstance(idx, int):
                    node_to_idx[node.name] = idx
            elif node.target == operator.add:
                input_indices = []
                for arg in node.args:
                    if isinstance(arg, torch.fx.node.Node) and arg.name in node_to_idx:
                        input_indices.append(node_to_idx[arg.name])
                if input_indices == [0, 1]:
                    layer_cfg = {
                        'type': 'element_sum',
                        'input_index': input_indices
                    }
                    record_layer(layer_cfg)
                else:
                    raise ValueError(
                    "MNSIM-2.0 only supports element sum of form \n" +
                    """ def forward(self, x):
                                return x[0] + x[1]
                            \n"""+
                    "See search_spaces.py class ElementSum")
            elif node.target == torch.nn.functional.max_pool2d:
                assert len(node.args) == 1
                layer_cfg = ({
                    'type': 'pooling',
                    'mode': 'MAX',
                    'kernel_size': node.args[1]
                })

                # unsafe to handle it like this...
                if  len(node.args) == 2:
                    stride = node.args[2]
                else:
                    raise ValueError("MNSIM requires stride for max pool")

                if  len(node.args) == 3:
                    pad =node.args[2]
                else:
                    pad = 0
                lay
                layer_cfg['padding'] = pad
                record_layer(layer_cfg)
            elif node.target == operator.mul:
                if(node.args[0].target == 'view' and node.args[1].target ==  operator.getitem):
                    
                    node_to_idx[node.args[0].name] = layer_idx
                    layer_idx += 1

                    base, idx = node.args[1].args
                    if base.op == 'placeholder' and isinstance(idx, int):
                        node_to_idx[node.name] = idx

                    input_indices = [node_to_idx[arg.name] for arg in node.args]
                    layer_cfg = {
                        'type': 'element_multiply',
                        'input_index': input_indices
                    }
                    record_layer(layer_cfg)
                else:
                    raise ValueError(
                    "MNSIM-2.0 only supports element multiplication of form \n" +
                    """def forward(self, x):
                            x[0] = x[0].view(x[0].shape[0], x[0].shape[1], 1, 1)
                            return x[0] * x[1] \n
                            """ +
                    "See search_spaces.py class MulView")
            elif node.target == torch.nn.functional.silu:
                layer_cfg = {'type': 'Swish'}
                record_layer(layer_cfg)
            elif node.target == torch.nn.functional.sigmoid:
                layer_cfg = {'type': 'Sigmoid'}
                record_layer(layer_cfg)
            elif node.target == torch.nn.functional.relu:
                layer_cfg = {'type': 'relu'}
                record_layer(layer_cfg)
            elif node.target.__name__ == 'flatten':
                layer_cfg = {'type': 'view'}
                record_layer(layer_cfg)
            elif node.target == torch.cat:
                if len(node.args) >= 2 and node.args[1] == 1:
                    layer_cfg = {
                        'type': 'concat',
                        'dim': 1,
                    }
                    record_layer(layer_cfg)
                    print("Concat was not validated fully. Best to avoid it")
                else:
                    raise ValueError("MNSIM-2.0 only allows concat layer with dim 1.")
            else:
                print("no match for", node, node.target, "in call_function")
        elif node.op == 'call_module':
            mod = dict(model.named_modules())[node.target]
            if isinstance(mod, nn.Conv2d) and mod.groups == mod.in_channels and mod.in_channels == mod.out_channels:
                if isinstance(mod.kernel_size, tuple) and mod.kernel_size[0] != mod.kernel_size[1] :
                    raise ValueError( "MNSIM 2.0 only supports quadratic kernels")
                layer_cfg = {
                    'type': 'conv',
                    'in_channels': mod.in_channels,
                    'out_channels': mod.out_channels,
                    'kernel_size': mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size,
                    'padding': mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding,
                    'stride': mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride,
                    'depthwise': 'separable'
                }
                if mod.bias is not None:
                    print("WARNING: Due to quantization MNSIM 2.0 does not support bias in convolution layer. Please consider using conv_bn2. Bias in convolution layer will be ignored.")
                record_layer(layer_cfg)
            # handle like normal conv2?
            elif isinstance(mod, nn.Conv2d):
                if isinstance(mod.kernel_size, tuple) and mod.kernel_size[0] != mod.kernel_size[1]:
                    raise ValueError( "MNSIM 2.0 only supports quadratic kernels")
                layer_cfg = ({
                    'type': 'conv',
                    'in_channels': mod.in_channels,
                    'out_channels': mod.out_channels,
                    'kernel_size': mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size,
                    'padding':  mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
                })
                if mod.bias is not None:
                    print("WARNING: Due to quantization MNSIM 2.0 does not support bias in convolution layer. Please consider using conv_bn2. Bias in convolution layer will be ignored.")
                record_layer(layer_cfg)
            elif isinstance(mod, nn.Linear):
                out_features = num_classes if (num_classes and mod.out_features == num_classes) else mod.out_features
                layer_cfg = ({
                    'type': 'fc',
                    'in_features': mod.in_features,
                    'out_features': out_features
                })
                record_layer(layer_cfg)
            elif isinstance(mod, nn.MaxPool2d):
                if isinstance(mod.kernel_size, tuple) and mod.kernel_size[0] != mod.kernel_size[1]:
                    raise ValueError( "MNSIM 2.0 only supports quadratic kernels")
                
                layer_cfg = ({
                    'type': 'pooling',
                    'mode': 'MAX',
                    'kernel_size': mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size,
                    'stride':  mod.stride if isinstance(mod.stride, int) else mod.stride[0],
                })
                if isinstance(mod, nn.MaxPool2d) and hasattr(mod, 'padding'):
                    pad = mod.padding
                    layer_cfg['padding'] = pad if isinstance(pad, int) else pad[0]
                record_layer(layer_cfg)
            elif isinstance(mod, nn.AdaptiveAvgPool2d):
                output_size = mod.output_size
                if output_size == (1, 1):
                    layer_cfg = {
                        'type': 'pooling',
                        'mode': 'ADA',
                        # Placeholder values? They still get used for latency modeling? Intentional?
                        'kernel_size': 1,
                        'stride': 1,
                    }
                    record_layer(layer_cfg)
                else:
                    raise ValueError (f"Unsupported AdaptiveAvgPool2d output size: {output_size}. MNSIM-2.0 only supports nn.AdaptiveAvgPool2d((1,1))")
            elif isinstance(mod, nn.AvgPool2d):
                if isinstance(mod.kernel_size, tuple) and mod.kernel_size[0] != mod.kernel_size[1]:
                    raise ValueError( "MNSIM 2.0 only supports quadratic kernels")
                layer_cfg = ({
                    'type': 'pooling',
                    'mode': 'AVE',
                    'kernel_size': mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0],
                    'kernel_size': mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0], 
                    'stride':  mod.stride if isinstance(mod.stride, int) else mod.stride[0],
                })
                if hasattr(mod, 'padding'):
                    pad = mod.padding
                else:
                    pad = 0
                layer_cfg['padding'] = pad if isinstance(pad, int) else pad[0]
                record_layer(layer_cfg)
            elif isinstance(mod, nn.Dropout):
                layer_cfg = {'type': 'dropout'}
                # if isinstance(node.args[0], torch.fx.node.Node):
                #     prev_node = node.args[0]
                #     if prev_node.name in node_to_idx:
                #         layer_cfg['input_index'] = [node_to_idx[prev_node.name]]
                record_layer(layer_cfg)

            elif isinstance(mod, torch.nn.BatchNorm1d):
                raise NotImplementedError("MNSIM 2.0 only supports BatchNorm2d")
                layer_cfg = {
                    'type':         'batch_norm_1d',
                    'num_features': mod.num_features,
                    'eps':          mod.eps,
                    'momentum':     mod.momentum,
                    'affine':       mod.affine
                }
                record_layer(layer_cfg)

            elif isinstance(mod, torch.nn.BatchNorm2d):
                layer_cfg = {
                    'type': 'bn',
                    'features': mod.num_features,
                    'eps': mod.eps,
                    'momentum': mod.momentum,
                    'affine': mod.affine
                }
                record_layer(layer_cfg)
                print ("Note: MSIM 2.0 only uses parameter 'features' for BatchNorm2D. Default for everything else.")
            elif isinstance(mod, torch.nn.BatchNorm3d):
                raise NotImplementedError("MNSIM 2.0 only supports BatchNorm2d")
                layer_cfg = {
                    'type':         'batch_norm_3d',
                    'num_features': mod.num_features,
                    'eps':          mod.eps,
                    'momentum':     mod.momentum,
                    'affine':       mod.affine
                }
                record_layer(layer_cfg)

            elif isinstance(mod, nn.SiLU):
                layer_cfg = {'type': 'Swish'}
                record_layer(layer_cfg)
            elif(isinstance(mod, nn.ReLU)):
                layer_cfg = {'type': 'relu'}
                record_layer(layer_cfg)
            elif(isinstance(mod, nn.Flatten)):
                layer_cfg = {'type': 'view'}
                record_layer(layer_cfg)
            elif isinstance(mod, nn.Sigmoid):
                layer_cfg = {'type': 'Sigmoid'}
                record_layer(layer_cfg)
            else:
                print("no match for", node, mod, node.target, "in call_modules")
        elif node.op == 'output':
            pass
        else:
            print("no match for", node, node.target)
    return layer_config_list
