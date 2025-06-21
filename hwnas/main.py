from collections import Counter, defaultdict
from datetime import datetime
import json
import os
import nni
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import cifar10net
from search_space import VGG8ModelSpaceCIFAR10
from nni.nas.evaluator import FunctionalEvaluator
import nni.nas.strategy as strategy
from nni.nas.experiment import NasExperiment
from nni.nas.experiment.config import NasExperimentConfig
from evaluator import hw_evaluation_model
from torch.utils.data import DataLoader
from hardware_aware_performance_estimation import get_hardware_metrics
from nni.mutable import Categorical
from nni.nas.nn.pytorch import LayerChoice
import torch

torch.set_float32_matmul_precision('medium')

def simple_test():
    df = pd.read_csv("hwnas/genome_dataset.csv", index_col=0)
    n_splits = 1
    group = "genome_id"
    batch_size = 256
    k = 7
    num_workers = 4

    if group is None:
        train_subset, val_subset = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    else:
        gss = GroupShuffleSplit(n_splits=n_splits, random_state=42, test_size=None, train_size=0.7)
        for i, (train_index, val_index) in enumerate(gss.split(df, df["label"], groups=df[group])):
            assert n_splits > i
            train_subset = df.iloc[train_index]
            val_subset = df.iloc[val_index]

    train_dataset = cifar10net.FCGRDataset(train_subset ,k)
    val_dataset = cifar10net.FCGRDataset(val_subset, k)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = cifar10net.CIFAR10NetFCGRNonSeq(num_classes=100, k=7)

    get_hardware_metrics(model=model, train_loader=train_loader, test_loader=val_loader)

def combine_model_dict(best_candidates):
    allowed_values = defaultdict(set)
    value_counts = defaultdict(Counter)
    lenght = len(best_candidates)

    for cfg in best_candidates:
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

# @profile
def main():

    batch_size = 128
    num_workers = 0
    max_epochs = 30
    max_steps = 1
    fast_dev_run = True
    torch.set_float32_matmul_precision('medium')

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    # Define a simple transform

    # full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # # Training set and loader
    # # train_dataset, val_dataset = random_split(full_train, [45000, 5000])
    # # train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # # val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # train_loader  = DataLoader(full_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # # Test set and loader
    # test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # search_strategy = strategy.DARTS() # mutation_hooks=[MixedConv2d.mutate, MixedLinear.mutate]
    # # evaluator = ClassificationModule(train_dataloaders=train_loader, val_dataloaders=test_loader, num_classes=10)
    # evaluator = Lightning(ClassificationModule(num_classes=10), Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, fast_dev_run=fast_dev_run, max_steps=max_steps), train_dataloaders=train_loader, val_dataloaders=test_loader)
   
    # # config = NasExperimentConfig("sequential", "simplified", "local", **{"debug":True})
    # config = NasExperimentConfig.default(model_space=model_space, evaluator=evaluator, strategy=search_strategy)
    # config.model_format = RawModelFormatConfig()
    # exp = NasExperiment(model_space, evaluator, search_strategy, config)
    # exp.config.max_trial_number = 25   
    # exp.config.trial_concurrency = 1 
    # exp.config.trial_gpu_number = 0
    # exp.config.execution_engine.name = "sequential"
    # exp.run(port=8081, debug = True)

    # model_space = VGG8ModelSpaceCIFAR10OneShot()
    # restrict_model_space(model_space, exp.export_top_models(formatter="dict", top_k=100))

    model_space = VGG8ModelSpaceCIFAR10()
    
    # model_space = TutorialModelSpace()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    search_strategy = strategy.TPE()

    evaluator = FunctionalEvaluator(hw_evaluation_model, **{"num_classes" : 10, "batch_size" : batch_size, "epochs" : max_epochs, "num_workers" : num_workers, "output_csv" : os.path.join(output_dir, "results.csv")})
    config = NasExperimentConfig("sequential", "simplified", "local", **{"debug":True})
    exp = NasExperiment(model_space, evaluator, search_strategy, config, id="5ylx1uk9")
    exp.config.max_trial_number = 50
    exp.config.execution_engine.name = "sequential"
    # exp.run(port=8081, debug = True)
    if exp.has_checkpoint():
        exp.resume()
    else:
        exp.run()
    tmp = exp.export_top_models(formatter="dict", top_k=50)
    with open(os.path.join(output_dir, "top_models.json"), "w") as f:
        json.dump(tmp, f, indent=4)

    with open(os.path.join(output_dir, "search_stat_dict.json"), "w") as f:
        json.dump(search_strategy.state_dict(), f, indent=4)

    with open(os.path.join(output_dir, "all_models.json"), "w") as f:
        json.dump(search_strategy.list_models(), f, indent=4)

if __name__ == "__main__":
    main()
