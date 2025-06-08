import math
import os
import time
import nni
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from nni.nas.space import ExecutableModelSpace
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm


sys.path.insert(0, '/mnt/c/Users/Luis/Documents/Uni-DESKTOP-F7N3QC8/TU Dresden/4. Semester/CC-Seminar/MNSIM-2.0')

from cifar10net import FCGRDataset
import util
from hardware_checker import get_hardware_metrics

def constrained_objective(accuracy, latency, energy, area,
                          max_latency, max_energy, max_area):
    if latency > max_latency or energy > max_energy or area > max_area:
        return -float('inf')  # discard
    return accuracy

def multiple_objective_function(accuracy, latency, energy, area,
                       alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    
    # Normalisation, take a look into the other papers?
    return alpha * accuracy - (beta * latency + gamma * energy + delta * area)

def objective(acc, latency, energy, area,
              w_acc=1.0, w_lat=0.1, w_en=0.05, A_max=1e8):
    """
    Returns a scalar F to maximize, without explicit reference values.
    If area > A_max, returns -inf to mark infeasible.
    """
    if area > A_max:
        return float("-inf")
    return w_acc * acc \
           - w_lat * math.log(latency) \
           - w_en * math.log(energy)


def train_epoch(model, device, train_loader, optimizer):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    
    for _, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
          correct, len(test_loader.dataset), accuracy))

    return accuracy

def hw_evaluation_model(model, group, num_classes, filepath="hwnas/genome_dataset.csv", nsplits = 1, k=7,  batch_size=64, num_workers = 4, epochs=30, lr=0.001):
    df = pd.read_csv(filepath, index_col=0)
    n_splits = 1
    # group = "genome_id"
    start = time.time()

    util.replace_conv_bias_with_bn(module=model)
    
    csv_path = "result.csv"
    file_exists = os.path.isfile(csv_path)

    # model = model_bn
    model_exec = ExecutableModelSpace(model)
    context = model_exec.status._frozen_context


    # if group is None:
    #     train_subset, val_subset = train_test_split(df, df["label"])
    # else:
    #     gss = GroupShuffleSplit(n_splits=n_splits, random_state=42, test_size=None, train_size=0.7)
    #     for i, (train_index, val_index) in enumerate(gss.split(df, df["label"], groups=df[group])):
    #         assert n_splits > i
    #         train_subset = df.iloc[train_index]
    #         val_subset = df.iloc[val_index]

    # train_dataset = FCGRDataset(train_subset ,k)
    # val_dataset = FCGRDataset(val_subset, k)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Define a simple transform
    transform = transforms.ToTensor()

    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Training set and loader
    train_dataset, val_dataset = random_split(full_train, [45000, 5000])
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Test set and loader
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    df = pd.DataFrame()
    if file_exists:
        df = pd.read_csv(csv_path, )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Evaluate on", device)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_metric = -np.inf
    patience = 8
    counter = 0
    min_delta = 1

    # return get_hardware_metrics(model=model, train_loader=train_loader, val_loader=val_loader) 

    # for epoch in range(epochs):
    #     print("Epoch", epoch)
    #     train_epoch(model, device, train_loader, optimizer)
    #     accuracy = test_epoch(model, device, val_loader)
    #     val_metric = accuracy
    #     if counter >= patience:
    #         print("Early stopping triggered")
    #         break
    #     if val_metric > best_metric + min_delta:
    #         best_metric = val_metric
    #         counter = 0
    #         print("Reset Patience")
    #     else:
    #         counter += 1
        # nni.report_intermediate_result(accuracy)
    accuracy, latency, energy, area = get_hardware_metrics(model=model, train_loader=train_loader, test_loader=test_loader, val_loader=val_loader, num_classes=num_classes)
    print("Evaluation done.")
    metric = objective(acc=accuracy, latency=latency, energy=energy, area=area)
    print("Final metric", metric)
    nni.report_final_result(metric=metric)
    result = pd.DataFrame([{"context": context, "accuracy" : accuracy, "latency" : latency, "energy" : energy , "area" : area}]) 
    
    
    result.to_csv(
        csv_path,
        mode='a' if file_exists else 'w',
        header=not file_exists,
        index=False
    )

    if file_exists:
        return
    end = time.time()
    print("Total time for", context, ":", end - start)


def evaluate_model(model, group, filepath="genome_dataset.csv", nsplits = 1, k=7, batch_size=64, num_workers = 4, epochs=30, lr=0.001):
    df = pd.read_csv(filepath, index_col=0)
    n_splits = 1
    group = "genome_id"

    if group is None:
        train_subset, val_subset = train_test_split(df, df["label"])
    else:
        gss = GroupShuffleSplit(n_splits=n_splits, random_state=42, test_size=None, train_size=0.7)
        for i, (train_index, val_index) in enumerate(gss.split(df, df["label"], groups=df[group])):
            assert n_splits > i
            train_subset = df.iloc[train_index]
            val_subset = df.iloc[val_index]

    train_dataset = FCGRDataset(train_subset ,k)
    val_dataset = FCGRDataset(val_subset, k)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(model)
    print(util.model_to_layer_config(model=model))
    
    best_metric = -np.inf
    patience = 8
    counter = 0
    min_delta = 1 

    for epoch in range(epochs):
        print("Epoch", epoch)
        train_epoch(model, device, train_loader, optimizer)
        accuracy = test_epoch(model, device, val_loader)
        val_metric = accuracy
        if counter >= patience:
            print("Early stopping triggered")
            break
        if val_metric > best_metric + min_delta:
            best_metric = val_metric
            counter = 0
            print("Reset Patience")
        else:
            counter += 1
        nni.report_intermediate_result(accuracy)
    print("Evaluation done.")

    nni.report_final_result(accuracy)



def evaluate_model_tutorial(model):
    # By v3.0, the model will be instantiated by default.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST('data/mnist', download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST('data/mnist', download=True, train=False, transform=transf), batch_size=64)


    for epoch in range(3):
        print("Epoch", epoch)
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer)
        # test the model for one epoch
        accuracy = test_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict

    # report final test result
    nni.report_final_result(accuracy)