import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import cifar10net
from search_space import CompatibleModelSpace
from nni.nas.evaluator import FunctionalEvaluator
import nni.nas.strategy as strategy
from nni.nas.experiment import NasExperiment
from nni.nas.experiment.config import NasExperimentConfig
from evaluator import hw_evaluation_model
from torch.utils.data import DataLoader
from hardware_checker import test

class CustomModel128(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input has 1 channel now instead of 3
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # → (6, 124, 124)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial dimensions
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # → (16, 58, 58) after pool → (16, 29, 29)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5) # → (120, 25, 25) after second pool
        self.dropout = nn.Dropout()
        # After conv3 + pool: 120 channels × 25 × 25 = 120 * 625 = 75000 features
        self.fc1 = nn.Linear(120 * 25 * 25, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # Input: (1,128,128) → (6,124,124)
        x = self.pool(x)            # → (6,62,62)

        x = F.relu(self.conv2(x))   # → (16,58,58)
        x = self.pool(x)            # → (16,29,29)

        x = F.relu(self.conv3(x))   # → (120,25,25)
        # x = self.pool(x)            # → (120,12,12)  ← Note: pooling again reduces 25→12 (floor)

        x = torch.flatten(x, 1)     # → (batch_size, 120 * 12 * 12 = 17280)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def simple_test():
    df = pd.read_csv("hwnas/genome_dataset.csv", index_col=0)
    n_splits = 1
    group = "genome_id"
    batch_size = 128
    k = 7
    num_workers = 4

    if group is None:
        train_subset, val_subset = train_test_split(df, df["label"])
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
    model = cifar10net.CIFAR10Net(num_classes=100)
    test(model=model, train_loader=train_loader, val_loader=val_loader)

def main():
    # search_strategy = strategy.Random() 
    # model_space = SimpleCIFAR10ModelSpace()
    # evaluator = FunctionalEvaluator(evaluate_model_tutorial, **{"group" : "genome_id", "batch_size" : 64, "epoch" : 3})

#     config = NasExperimentConfig(
#     execution_engine = 'ts',
#     model_format='simplified',
#     trial_gpu_number=1,
#     trial_concurrency=1,
#     training_service={
#         'platform': 'local',
#         'useActiveGpu': True,
#         'gpuIndices': [0],
#         'maxTrialNumberPerGpu': 1
#     },
#     max_trial_number=50,
#     max_experiment_duration='2h'
# )


    simple_test()
    return 
    model_space = CompatibleModelSpace()
    search_strategy = strategy.RegularizedEvolution(population_size=10, sample_size=3)
    evaluator = FunctionalEvaluator(hw_evaluation_model, **{"group" : "genome_id", "batch_size" : 256, "epochs" : 1})
    config = NasExperimentConfig("sequential", "simplified", "local")
    exp = NasExperiment(model_space, evaluator, search_strategy, config)
    exp.config.execution_engine.name = "sequential"
    exp.config.debug = True
    nni.enable_global_logging(True)
    exp.run(port=8081)

if __name__ == "__main__":
    main()
