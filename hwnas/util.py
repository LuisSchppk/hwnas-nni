import operator
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
import yaml

from search_space import DepthwiseSeparableConv

def model_to_layer_config(model, filepath=None, num_classes=None):
    """
    Parse a PyTorch model to a layer config list and save to YAML.
    Supports conv, relu, pooling, view(flatten), dropout, fc layers.
    """
    layer_config_list = []

    for layer in model.modules():
        # Skip the top-level model container itself
        if layer == model:
            continue

        if isinstance(layer, nn.Conv2d):
            layer_config_list.append({
                'type': 'conv',
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size,
                'stride': layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride,
            })

        elif isinstance(layer, nn.ReLU):
            layer_config_list.append({'type': 'relu'})

        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            mode = 'MAX' if isinstance(layer, nn.MaxPool2d) else 'AVG'
            layer_config_list.append({
                'type': 'pooling',
                'mode': mode,
                'kernel_size': layer.kernel_size,
                'stride': layer.stride,
            })

        elif isinstance(layer, nn.Flatten):
            layer_config_list.append({'type': 'view'})

        elif isinstance(layer, nn.Dropout):
            layer_config_list.append({'type': 'dropout'})

        elif isinstance(layer, nn.Linear):
            out_features = num_classes if (num_classes and layer.out_features == num_classes) else layer.out_features
            layer_config_list.append({
                'type': 'fc',
                'in_features': layer.in_features,
                'out_features': out_features,
            })
        else:
            print("Unrequired information", layer)

    if filepath is not None:
        with open(filepath, 'w') as f:
            yaml.dump(layer_config_list, f)

    return layer_config_list


def trace_model(model, input_shape, num_classes=None):
    dummy_input = torch.randn(*input_shape)
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
        if node.op == 'call_module':
            mod = dict(model.named_modules())[node.target]


            if isinstance(mod, DepthwiseSeparableConv):
                # mod.depthwise and mod.pointwise are the two sub‐convs
                # We emit a single entry with the ‘depthwise’ flag
                layer_cfg = {
                    'type': 'conv',
                    'in_channels': mod.depthwise.in_channels,
                    'out_channels': mod.pointwise.out_channels,
                    'kernel_size': mod.depthwise.kernel_size[0],
                    'padding': mod.depthwise.padding[0] if isinstance(mod.depthwise.padding, tuple) else mod.depthwise.padding,
                    'stride': mod.depthwise.stride[0] if isinstance(mod.depthwise.stride, tuple) else mod.depthwise.stride,
                    'depthwise': 'separable'
                }
                record_layer(layer_cfg)
            elif isinstance(mod, nn.Conv2d):
                layer_cfg = ({
                    'type': 'conv',
                    'in_channels': mod.in_channels,
                    'out_channels': mod.out_channels,
                    'kernel_size': mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size,
                    'padding':  mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
                })

                # node.args input to node?
                # if isinstance(node.args[0], torch.fx.node.Node):
                #     prev_node = node.args[0]
                #     if prev_node.name in node_to_idx:
                #         layer_cfg['input_index'] = [node_to_idx[prev_node.name]]
                record_layer(layer_cfg)


            elif isinstance(mod, nn.Linear):
                out_features = num_classes if (num_classes and mod.out_features == num_classes) else mod.out_features
                layer_cfg = ({
                    'type': 'fc',
                    'in_features': mod.in_features,
                    'out_features': out_features
                })

                # if isinstance(node.args[0], torch.fx.node.Node):
                #     prev_node = node.args[0]
                #     if prev_node.name in node_to_idx:
                #         layer_cfg['input_index'] = [node_to_idx[prev_node.name]]
                record_layer(layer_cfg)


            elif isinstance(mod, nn.MaxPool2d):
                layer_cfg = ({
                    'type': 'pooling',
                    'mode': 'MAX',
                    'kernel_size': mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0],
                    'stride':  mod.stride if isinstance(mod.stride, int) else mod.stride[0],
                })

                if isinstance(mod, nn.MaxPool2d) and hasattr(mod, 'padding'):
                    # If padding is non-zero, include it
                    pad = mod.padding
                    layer_cfg['padding'] = pad if isinstance(pad, int) else pad[0]
                
                # Track input index
                # if isinstance(node.args[0], torch.fx.node.Node):
                #     prev_node = node.args[0]
                #     if prev_node.name in node_to_idx:
                #         layer_cfg['input_index'] = [node_to_idx[prev_node.name]]
                record_layer(layer_cfg)

            elif isinstance(mod, nn.AdaptiveAvgPool2d):
                # Estimate kernel and stride if possible
                output_size = mod.output_size
                if output_size == (1, 1):
                    layer_cfg = {
                        'type': 'pooling',
                        'mode': 'ADA',
                        # Placeholder values (not used by ADA in MNSIM-2.0)
                        'kernel_size': None,
                        'stride': None
                    }

                    # if isinstance(node.args[0], torch.fx.node.Node):
                    #     prev_node = node.args[0]
                    #     if prev_node.name in node_to_idx:
                    #         layer_cfg['input_index'] = [node_to_idx[prev_node.name]]
                    record_layer(layer_cfg)
                else:
                    raise ValueError (f"Unsupported AdaptiveAvgPool2d output size: {output_size}. MNSIM-2.0 only supports nn.AdaptiveAvgPool2d((1,1))")
        
            elif isinstance(mod, nn.Dropout):
                layer_cfg = {'type': 'dropout'}
                # if isinstance(node.args[0], torch.fx.node.Node):
                #     prev_node = node.args[0]
                #     if prev_node.name in node_to_idx:
                #         layer_cfg['input_index'] = [node_to_idx[prev_node.name]]
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
            elif(isinstance(mod, nn.SiLU)):
                layer_cfg = {'type': 'Swish'}
                record_layer(layer_cfg)

                
            elif node.target == operator.add:
                    # Collect input nodes involved in the addition
                    input_indices = []
                    for arg in node.args:
                        if isinstance(arg, torch.fx.node.Node) and arg.name in node_to_idx:
                            input_indices.append(node_to_idx[arg.name])
                    layer_cfg = {
                        'type': 'element_sum',
                        'input_index': input_indices
                    }
                    record_layer(layer_cfg)

            elif node.target == operator.mul:
                    input_indices = [node_to_idx[arg.name] for arg in node.args]
                    layer_cfg = {
                        'type': 'element_multiply',
                        'input_index': input_indices
                    }
                    record_layer(layer_cfg)

            elif node.target == torch.nn.functional.silu:
                    layer_cfg = {'type': 'Swish'}
                    if isinstance(node.args[0], torch.fx.node.Node):
                        layer_cfg['input_index'] = [node_to_idx[node.args[0].name]]
                    
            else:
                    print("no match for", mod, node, node.target)
    return layer_config_list
