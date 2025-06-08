

from collections import OrderedDict
import re

from torch import Tensor


def build_tree(state_dict):
    tree = OrderedDict()
    for key, tensor in state_dict.items():
        parts = key.split('.')
        node = tree

        if len(parts) == 2:
            node[".".join(parts)] = tensor
        else:    
            for p in parts[:-2]:
                node = node.setdefault(p, {})
            node[".".join(parts[-2:])] = tensor
    return tree

def get_leaves(nodes):
    leaves = []
    for key, child in nodes.items():
        if not isinstance(child, dict):
            leaves.extend([(key, child, 0)])
        else:
            leaves.extend(get_leaves_helper(child, 0))
    return leaves

def get_leaves_helper(node, depth):
    leaves = []
    for key, child in node.items():
        if not isinstance(child, dict):
            leaves.extend([(key, child, depth)])
        else:
            leaves.extend(get_leaves_helper(child, depth+1))
    return leaves

def increase_idx(key):
    match = re.match(r'([a-zA-Z]+)(\d+)\.(.+)', key)
    if match:
        name, idx, attr = match.groups()
        new_idx = int(idx) + 1
        return name + str(new_idx) + "." + attr
    else:
        raise ValueError("Can only parse layer keys of from <name><idx>.<attr>")

def process(leaves):
    seen_keys = []
    new_state_dict = OrderedDict()
    for key, value, _ in leaves:
        while key in seen_keys:
            key = increase_idx(key)
        seen_keys.append(key)
        new_state_dict.update({key: value})
    return new_state_dict

def flatten_dict(state_dict):
    tree = build_tree(state_dict)
    inner_nodes = [(k, v) for k, v in tree.items() if isinstance(v, dict)]
    leaves = get_leaves(tree)
    flattend_dict = process(leaves)
    return flattend_dict




