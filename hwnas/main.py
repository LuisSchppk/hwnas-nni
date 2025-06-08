import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import cifar10net
from search_space import TutorialModelSpace
from nni.nas.hub.pytorch import DARTS as DartsSpace
from nni.nas.evaluator import FunctionalEvaluator
import nni.nas.strategy as strategy
from nni.nas.experiment import NasExperiment
from nni.nas.experiment.config import NasExperimentConfig
from evaluator import hw_evaluation_model
from torch.utils.data import DataLoader
from hardware_checker import get_hardware_metrics

def simple_test():
    df = pd.read_csv("hwnas/genome_dataset.csv", index_col=0)
    n_splits = 1
    group = "genome_id"
    batch_size = 128
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
    get_hardware_metrics(model=model, train_loader=train_loader, val_loader=val_loader)


def main():
    model_space = TutorialModelSpace()
    search_strategy = strategy.RegularizedEvolution(population_size=10, sample_size=3)
    evaluator = FunctionalEvaluator(hw_evaluation_model, **{"filepath":"hwnas/genome_dataset.csv", "group" : "genome_id", "batch_size" : 256, "epochs" : 1, "num_workers" : 8})
    config = NasExperimentConfig("sequential", "simplified", "local", **{"debug":True})
    exp = NasExperiment(model_space, evaluator, search_strategy, config)
    exp.config.max_trial_number = 25   
    exp.config.trial_concurrency = 1 
    exp.config.trial_gpu_number = 0
    exp.config.execution_engine.name = "sequential"
    exp.run(port=8081, debug = True)

if __name__ == "__main__":
    main()
