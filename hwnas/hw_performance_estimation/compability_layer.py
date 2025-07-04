import operator
import torch
import torch.nn as nn
from torch.fx import symbolic_trace

def translate_state_dict_structure_file(
    state_dict, quantize_config_list, structure_file, tmp
):

    layers = list({key.split(".")[0] for key in state_dict.keys()})

    # print(structure_file)
    layer_prefix_list = [
        ("".join(filter(str.isalpha, p)), "".join(filter(str.isdigit, p)))
        for p in layers
    ]

    map = dict()
    for type, suffix in layer_prefix_list:
        count = 1
        for idx, layer_info in enumerate(structure_file):
            if layer_info.get("type") == type:
                if count == int(suffix):
                    map.update({type + suffix: idx})
                    # print("MAP", type, suffix, "to", layer_info[0][0].get("type"), structure_file.index(layer_info))
                    break
                else:
                    count += 1
    adapted_state_dict = {}

    for key, _ in state_dict.items():
        if key.count(".") > 1:
            print("WARNING", key)

        key_first, key_last = key.split(".")

        if key_last == "weight" and ("conv" in key_first or "fc" in key_first):
            adapted_state_dict.update(
                {
                    "layer_list."
                    + str(map.get(key_first, key_first))
                    + ".layer_list.0."
                    + key_last: _
                }
            )
        elif "conv" in key_first and "bias" in key:
            print(
                "Due to quantization MNSIM-2.0 does not support bias in convolution layer. Please use conv2d + batchnorm insead."
            )
            continue
        elif "fc" in key_first and "bias" in key:
            print(
                "Due to quantization MNSIM-2.0 does not support bias in fc layer. Please use fc + batchnorm insead."
            )
            continue
        else:
            adapted_state_dict.update(
                {
                    "layer_list."
                    + str(map.get(key_first, key_first))
                    + "."
                    + "layer"
                    + "."
                    + key_last: _
                }
            )

        quantize_config = quantize_config_list[(map.get(key_first))]
        # Only read and same value for all?
        bit_scale = torch.FloatTensor(
            [
                [quantize_config["activation_bit"], -1],
                [quantize_config["weight_bit"], -1],
                [quantize_config["activation_bit"], -1],
            ]
        )

        last_value = (-1) * torch.ones(1)
        if "bn" not in key_first:
            adapted_state_dict.update(
                {
                    "layer_list."
                    + str(map.get(key_first, key_first))
                    + ".bit_scale_list": bit_scale
                }
            )
        adapted_state_dict.update(
            {
                "layer_list."
                + str(map.get(key_first, key_first))
                + ".last_value": last_value
            }
        )

    for idx, _ in enumerate(structure_file):
        if idx not in map.values():

            last_value = (-1) * torch.ones(1)
            adapted_state_dict.update(
                {"layer_list." + str(idx) + ".last_value": last_value}
            )

    return adapted_state_dict

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