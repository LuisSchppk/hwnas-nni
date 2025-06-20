import configparser
import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm

from MNSIM.Interface import quantize
sys.path.insert(0, '/mnt/c/Users/Luis/Documents/Uni-DESKTOP-F7N3QC8/TU Dresden/4. Semester/CC-Seminar/MNSIM-2.0')


import torch
import torch.profiler


import MNSIM
from util import trace_model
from MNSIM.Accuracy_Model.Weight_update import weight_update
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Energy_Model.Model_energy import Model_energy
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from mnsim_wrapper import WrappedTestTrainInterface

from torch.optim import Adam
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from translate_state_dict import *

def get_hardware_metrics(model, train_loader, test_loader, val_loader, num_classes, max_epochs = 30, device = "cuda"):
    # return 1, 1, 1, 1
    inputs, _ = next(iter(train_loader))
    input_shape = inputs.shape
    
    layer_config_list = trace_model(model=model, input_shape=input_shape, num_classes=num_classes)

    # layer_config_list = []
    # layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
    # layer_config_list.append({'type': 'bn', 'features': 128})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
    # layer_config_list.append({'type': 'bn', 'features': 128})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
    # layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
    # layer_config_list.append({'type': 'bn', 'features': 256})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
    # layer_config_list.append({'type': 'bn', 'features': 256})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
    # layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
    # layer_config_list.append({'type': 'bn', 'features': 512})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
    # layer_config_list.append({'type': 'bn', 'features': 512})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
    # layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'padding': 0})
    # layer_config_list.append({'type': 'bn', 'features': 1024})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
    # layer_config_list.append({'type': 'view'})
    # layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': num_classes})
    
    network_module = None # not used in this version
    state_dict = flatten_dict(model.state_dict())
    # state_dict = None
    # state_dict = None
    train_loader = train_loader
    test_loader = test_loader
    sim_config_path = "/mnt/c/Users/Luis/Documents/Uni-DESKTOP-F7N3QC8/TU Dresden/4. Semester/CC-Seminar/MNSIM-2.0/SimConfig.ini"

    # config = configparser.ConfigParser()
    # config.optionxform = str 
    # config.read(sim_config_path)
    # sub_array_size = model.sub_array_size
    # config['Crossbar level']["Subarray_Size"]

    # if xbar_size < int(sub_array_size):
    #     sub_array_size = xbar_size
    #     print("REDUCED SUBARRAY SIZE TO", xbar_size, "to fit into crossbar")

    # config['Crossbar level']['Xbar_Size'] = f'{xbar_size}, {xbar_size}'
    # config['Crossbar level']["Subarray_Size"] = f'{sub_array_size}'

    # with open(sim_config_path, 'w') as configfile:
    #     config.write(configfile)

    enable_variation = False
    enable_SAF = True
    enable_R_ratio = False
    enable_fixed_Qrange = False

    test_train_interface = WrappedTestTrainInterface(layer_config_list=layer_config_list, network_module=network_module, train_loader=train_loader, test_loader=test_loader, sim_config_path=sim_config_path, input_shape=input_shape, state_dict=state_dict, device="cuda")
    # print("XBar size:", test_train_interface.xbar_row)
    structure_file = test_train_interface.get_structure()
    TCG_mapping = TCG(structure_file, sim_config_path)

    hardware_modeling_start_time = time.time()
    latency = Model_latency(NetStruct=structure_file, SimConfig_path=sim_config_path, TCG_mapping=TCG_mapping)
    latency.calculate_model_latency(mode=1)
    # print("========================Latency Results=================================")
    # latency.model_latency_output()
    total_latency =  max(max(latency.finish_time))
    print("Latency:", total_latency, "ns")

    area = Model_area(NetStruct=structure_file, SimConfig_path=sim_config_path, TCG_mapping=TCG_mapping)
    # area.calculate_model_area()
    # print("========================Area Results=================================")
    # area.model_area_output()
    total_area = area.arch_total_area
    print("Area:", total_area, "um^2")

    power = Model_inference_power(NetStruct=structure_file, SimConfig_path=sim_config_path, TCG_mapping=TCG_mapping)
    # print("========================Power Results=================================")
    # power.model_power_output()
    total_power = power.arch_total_power
    print("Power:", total_power, "W")

    energy = Model_energy(NetStruct=structure_file, SimConfig_path=sim_config_path,
                                TCG_mapping=TCG_mapping, model_latency=latency, model_power=power)
    # print("========================Energy Results=================================")
    # energy.model_energy_output()
    total_energy = energy.arch_total_energy
    print("Energy:",total_energy, "nJ")
    hardware_modeling_end_time = time.time()

    hardware_modeling_time = hardware_modeling_end_time - hardware_modeling_start_time
    print("Hardware modeling time:", hardware_modeling_time, "s")

    # Accuracy currently not possible. MNSIM-2.0 

    # print("======================================")
    # print("Accuracy simulation will take a few minutes on GPU")
    accuracy_modeling_start_time = time.time()
    update_weights(test_train_interface.net, train_loader=train_loader, device=device)
    weight = test_train_interface.get_net_bits()
    weight_2 = weight_update(sim_config_path, weight, is_Variation=enable_variation, is_SAF=enable_SAF, is_Rratio=enable_R_ratio)
    if not (enable_fixed_Qrange):
        print("Original accuracy:", origin_evaluate(net=test_train_interface.net, val_loader=val_loader, method='FIX_TRAIN', adc_action='SCALE'))

        cim_accuracy = eval_cim_accuracy(val_loader=val_loader, net=test_train_interface.net, net_bit_weights=weight_2, adc_action='SCALE')
        print("PIM-based computing accuracy:", cim_accuracy)
    else:
        print("Original accuracy:", origin_evaluate(net=test_train_interface.net, val_loader=val_loader, method='FIX_TRAIN', adc_action='FIX'))
        
        cim_accuracy  = eval_cim_accuracy(val_loader=val_loader, net=test_train_interface.net, net_bit_weights=weight_2, adc_action='FIX')
        print("PIM-based computing accuracy:", cim_accuracy)
    accuracy_modeling_end_time = time.time()
    accuracy_modeling_time = accuracy_modeling_end_time - accuracy_modeling_start_time
    print("Accuracy modeling time:", accuracy_modeling_time)

    return cim_accuracy, total_latency, total_energy, total_area

def eval_cim_accuracy(val_loader, net, net_bit_weights, adc_action = 'SCALE', device="cuda"):
    net.to(device)
    print("Eval on device:", device)
    net.eval()
    test_correct = 0
    test_total = 0
    net.compile()
    with torch.no_grad():
        for i, (data, labels) in enumerate(val_loader):
            if i > 10:
                break
            data = data.to(device)
            test_total += labels.size(0)
            outputs = net.set_weights_forward(data, net_bit_weights, adc_action)
               
            labels = labels.to(device)
            _, predicted = torch.max(outputs, 1)
            
            test_correct += (predicted == labels).sum().item() 
    return test_correct / test_total

def origin_evaluate(net, val_loader, method = 'SINGLE_FIX_TEST', adc_action = 'SCALE', device="cuda"):
    net.to(device)
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i > 10:
                break
            images = images.to(device)
            test_total += labels.size(0)
            
            outputs = net(images, method, adc_action)
            
            # predicted
            labels = labels.to(device)
            _, predicted = torch.max(outputs, 1)
            
            test_correct += (predicted == labels).sum().item()
    return test_correct / test_total


def update_weights(net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    lr: float = 1e-3,          
    weight_decay: float = 0.01):

    net = torch.nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    images, labels =  next(iter(train_loader))
    optimizer.zero_grad()
    images = images.to(device)
    labels = labels.to(device)

    outputs = net(images, 'FIX_TRAIN')

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

def train_net(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    dir_path: str,
    epochs: int = 60,
    lr: float = 1e-3,          
    weight_decay: float = 0.01,
    milestones: list[int] = [30, 60],
    gamma: float = 0.1,         
) -> None:
    net = torch.nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    best_metric = -np.inf
    patience = 8
    counter = 0
    min_delta = 0.01 

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images, 'FIX_TRAIN')

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1:3d}, Batch {i+1:3d}/{len(train_loader):3d}, Loss: {loss.item():.4f}", end='\r')

        # scheduler.step()
        val_metric =  eval_net(net, test_loader, epoch + 1, device)
        if counter >= patience:
            print("Early stopping triggered")
            torch.cuda.empty_cache()
            break
        if val_metric > best_metric + min_delta:
            best_metric = val_metric
            counter = 0
            print("Reset Patience")
        else:
            counter += 1

    torch.cuda.empty_cache()
    return eval_net(net, test_loader, epoch + 1, device)



def eval_net(
net: nn.Module,
test_loader: torch.utils.data.DataLoader,
epoch: int,
device: torch.device
) -> float:
    net.to(device)
    net.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images, 'FIX_TRAIN')

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    accuracy = test_correct / test_total if test_total > 0 else 0.0

    print(f"{time.asctime()}: After epoch {epoch}, accuracy is {accuracy:.4f}")
    return accuracy