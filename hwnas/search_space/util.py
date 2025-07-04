import nni
from nni.mutable import Categorical
from nni.nas.nn.pytorch import LayerChoice
from collections import Counter, defaultdict

def combine_model_dict(model_dicts):
    """
    Analyzes a list of architecture configuration dictionaries to compute:
    1. The set of allowed (unique) values for each key across all configurations.
    2. The empirical distribution (as relative frequencies) of each key's values.
    """
    allowed_values = defaultdict(set)
    value_counts = defaultdict(Counter)
    lenght = len(model_dicts)

    for cfg in model_dicts:
        for key, value in cfg.items():
            allowed_values[key].add(value)
            value_counts[key][value] += 1

    allowed_values = {k: sorted(v) for k, v in allowed_values.items()}

    value_distributions = {}
    for key, values in allowed_values.items():
        distribution = [value_counts[key][v] / lenght for v in values]
        value_distributions[key] = distribution
    return allowed_values, value_distributions

def restrict_model_space(model_space, best_candidates):
    """
    Restricts a model search space to only include the configurations (hyperparameter values)
    observed in a given list of best candidate configurations.
    """
    restricted_search_space_dict, value_distributions = combine_model_dict(best_candidates)
    mutable_keys =[]
    hardware_param = ["xbar_size"]
    for mut in model_space.mutables:
        if mut.label in hardware_param:
            mutable_keys.append(mut.label)
            continue
        elif isinstance(mut, Categorical) and restricted_search_space_dict.get(mut.label, None) is not None:
            old_values = set(mut.values)  # Save original
            new_values = set(restricted_search_space_dict[mut.label])
            removed = old_values - new_values
            mut.values = nni.choice(label=mut.label, choices=sorted(new_values))
            mutable_keys.append(mut.label)
            if removed:
                print(f"Removed from '{mut.label}': {sorted(removed)}")
        else:
            raise ValueError("Unknown Mutable", mut)

    for key, value in restricted_search_space_dict.items():
        if key in mutable_keys:
            continue
        layer_choice = getattr(model_space, key)
        assert isinstance(layer_choice, LayerChoice)
        assert len(layer_choice.mutables) == 1
        mut = layer_choice.mutables[0]
        assert restricted_search_space_dict.get(mut.label, None) is not None
        old_values = set(mut.values)
        new_values = set(restricted_search_space_dict[mut.label])
        removed = old_values - new_values
        mut.values = nni.choice(label=mut.label, choices=sorted(new_values))
        mut.weights = value_distributions[mut.label]
        if removed:
            print(f"Removed from '{mut.label}': {sorted(removed)}")