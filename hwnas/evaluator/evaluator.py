import ast
import math
import os
import time
import nni
import numpy as np
import pandas as pd
from nni.nas.space import ExecutableModelSpace
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import util
from hw_performance_estimation.hw_performance_estimation import get_hardware_metrics

def constrained_objective(
    accuracy, latency, energy, area, max_latency, max_energy, max_area
):
    if latency > max_latency or energy > max_energy or area > max_area:
        return -float("inf")  # discard
    return accuracy


def objective(acc, latency, energy, area, w_acc=1.0, w_lat=1.0, w_en=1.0, A_max=1e8):
    """
    Returns a scalar F to maximize, without explicit reference values.
    If area > A_max, returns -inf to mark infeasible.
    """
    # if area > A_max:
    #     return float("-inf")

    log_lat = math.log(latency, 10) if latency > 0 else np.inf
    log_eng = math.log(energy, 10) if energy > 0 else np.inf

    return w_acc * acc - w_lat * log_lat - w_en * log_eng


def train_epoch(model, device, train_loader, optimizer):
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    for _, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="Training"
    ):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


def test_epoch(model, device, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct, len(test_loader.dataset), accuracy
        )
    )

    return accuracy

def hw_evaluation_model(
    model,
    num_classes,
    output_csv,
    batch_size=64,
    num_workers=4,
    epochs=30,
    lr=0.001,
    device="cuda",
    csv_suffix="_current.",
    hardware_config = "MNSIM-2.0/SimConfig.ini",
):
    
    """
    Evaluate the given model on the hardware.
    """
    start = time.time()

    model.to(device)
    util.replace_conv_bias_with_bn(module=model, device=device)

    file_exists = os.path.isfile(output_csv)

    # get model state
    model_exec = ExecutableModelSpace(model)
    context = model_exec.status._frozen_context

    print("Model", context)

    start_time = time.time()
    df = pd.DataFrame()
    matching_rows = pd.DataFrame()

    # Check if file with presearched models exist.
    if file_exists:
        df = pd.read_csv(
            output_csv,
        )
        df["config_dict"] = df["context"].apply(ast.literal_eval)

        def matches_target(config_dict, target_dict):
            return all(config_dict.get(k) == v for k, v in target_dict.items())

        matching_rows = df[
            df["config_dict"].apply(lambda x: matches_target(x, context))
        ]

    # Check whether this model was already evaluated. (In a previous search using the same configuration).
    if not matching_rows.empty:
        print("Reload results for known configuration")
        first_row = matching_rows.iloc[0]
        accuracy = first_row["accuracy"]
        latency = first_row["latency"]
        energy = first_row["energy"]
        area = first_row["area"]
    else:

        # Load Cifar10 Training, Test and Validation data.
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        full_train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        train_dataset, val_dataset = random_split(full_train, [45000, 5000])
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Train model on host device.
        best_metric = -np.inf
        patience = 8
        counter = 0
        min_delta = 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

        # Hardware Evaluation.
        accuracy, latency, energy, area = get_hardware_metrics(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            max_epochs=epochs,
            sim_config=hardware_config,
            device=device
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate final metric
    metric = objective(
        acc=accuracy, latency=latency, energy=energy, area=area, w_acc=100
    )
    print(f"Evaluation done in {elapsed_time:.2f} seconds.")
    print("Final metric", metric)
    nni.report_final_result(metric=metric)
    result = pd.DataFrame(
        [
            {
                "context": context,
                "accuracy": accuracy,
                "latency": latency,
                "energy": energy,
                "area": area,
            }
        ]
    )

    # Store results in db for overall search space
    result.to_csv(
        output_csv,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )

    # Store results in db for current search
    output_csv2 = output_csv.replace(".", csv_suffix)
    result.to_csv(
        output_csv2,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )

    end = time.time()
    print("Total time for", context, ":", end - start)