import collections
import configparser
import math
import re
from MNSIM.Latency_Model.Model_latency import Model_latency
import util
from pickle import FALSE

import torch
import sys
sys.path.insert(0, '/mnt/c/Users/Luis/Documents/Uni-DESKTOP-F7N3QC8/TU Dresden/4. Semester/CC-Seminar/MNSIM-2.0')

import MNSIM
from MNSIM import Interface
import MNSIM
from MNSIM.Interface.interface import TrainTestInterface
from MNSIM.Interface.network import NetworkGraph
import bisect

def translate_state_dict_structure_file(state_dict, quantize_config_list, structure_file, tmp):

    layers = list({key.split('.')[0] for key in state_dict.keys()})

    # print(structure_file)
    layer_prefix_list = [
    (''.join(filter(str.isalpha, p)), ''.join(filter(str.isdigit, p)))
    for p in layers
    ]
    
    map =  dict()
    for type, suffix in layer_prefix_list:
        count = 1
        for idx, layer_info in enumerate(structure_file):
            if layer_info.get('type') == type:
                if count == int(suffix):
                    map.update({type + suffix: idx})
                    # print("MAP", type, suffix, "to", layer_info[0][0].get("type"), structure_file.index(layer_info))
                    break
                else:
                    count+=1
    adapted_state_dict = {}
 
    for key, _ in state_dict.items():
        if key.count(".") > 1:
            print("WARNING", key)
        
        key_first, key_last = key.split('.')

        if(key_last == "weight" and ("conv" in key_first or "fc" in key_first)):
            adapted_state_dict.update({"layer_list." + str(map.get(key_first, key_first)) + ".layer_list.0." + key_last: _})
        elif("conv" in key_first and "bias" in key):
            print("Due to quantization MNSIM-2.0 does not support bias in convolution layer. Please use conv2d + batchnorm insead.")
            continue
        elif("fc" in key_first and "bias" in key):
            print("Due to quantization MNSIM-2.0 does not support bias in fc layer. Please use fc + batchnorm insead.")
            continue
        else:
            adapted_state_dict.update({"layer_list." + str(map.get(key_first, key_first)) + "." + "layer" + "." + key_last: _})

        quantize_config= quantize_config_list[(map.get(key_first))]
        # Only read and same value for all?
        bit_scale = torch.FloatTensor([
                [quantize_config['activation_bit'], -1],
                [quantize_config['weight_bit'], -1],
                [quantize_config['activation_bit'], -1]
                       ])
        
        last_value = (-1) * torch.ones(1)
        if("bn" not in key_first):
            adapted_state_dict.update({"layer_list." + str(map.get(key_first, key_first)) + '.bit_scale_list' : bit_scale})
        adapted_state_dict.update({"layer_list." + str(map.get(key_first, key_first)) + ".last_value" : last_value})

    for idx, _ in enumerate(structure_file):
        if idx not in map.values():
        
            last_value = (-1) * torch.ones(1)
            adapted_state_dict.update({"layer_list." + str(idx) +".last_value" : last_value})
        
    return adapted_state_dict

class DatasetModuleDummy():
    def __init__(self, train_loader, val_loader):
        self.data_loaders = [train_loader, val_loader]

    def get_dataloader(self):
        return self.data_loaders

class WrappedTestTrainInterface(TrainTestInterface):
     def __init__(self, layer_config_list, network_module, train_loader, test_loader, sim_config_path, input_shape, state_dict, weights_file = None, device = None, extra_define = None):
        
        self.network_module = network_module # Unused
        self.dataset_module = DatasetModuleDummy(train_loader, test_loader)
        self.weights_file = weights_file # Unused
        self.test_loader = test_loader

        # load simconfig
        ## xbar_size, input_bit, weight_bit, ADC_quantize_bit
        xbar_config = configparser.ConfigParser()
        xbar_config.read(sim_config_path, encoding = 'UTF-8')
        self.hardware_config = collections.OrderedDict()

        # xbar_size
        xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.xbar_row = xbar_size[0]
        self.xbar_column = xbar_size[1]
        self.hardware_config['xbar_size'] = xbar_size[0]
        self.hardware_config['type'] = int(xbar_config.get('Process element level', 'PIM_Type'))
        self.hardware_config['xbar_polarity'] = int(xbar_config.get('Process element level', 'Xbar_Polarity'))
        self.hardware_config['DAC_num'] = int(xbar_config.get('Process element level', 'DAC_Num'))

        # device bit
        self.device_bit = int(xbar_config.get('Device level', 'Device_Level'))
        self.hardware_config['weight_bit'] = math.floor(math.log2(self.device_bit))

        # weight_bit means the weight bitwidth stored in one memory device
        # input bit and ADC bit
        ADC_choice = int(xbar_config.get('Interface level', 'ADC_Choice'))
        DAC_choice = int(xbar_config.get('Interface level', 'DAC_Choice'))
        temp_DAC_bit = int(xbar_config.get('Interface level', 'DAC_Precision'))
        temp_ADC_bit = int(xbar_config.get('Interface level', 'ADC_Precision'))
        ADC_precision_dict = {
            -1: temp_ADC_bit,
            1: 10,

            # reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
            2: 8,

            # reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
            3: 8,  # reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
            4: 6,  # reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
            5: 8,  # ASPDAC1
            6: 6,  # ASPDAC2
            7: 4,  # ASPDAC3
            8: 1,
            9: 6
        }
        DAC_precision_dict = {
            -1: temp_DAC_bit,
            1: 1,  # 1-bit
            2: 2,  # 2-bit
            3: 3,  # 3-bit
            4: 4,  # 4-bit
            5: 6,  # 6-bit
            6: 8,  # 8-bit
            7: 1
        }
        self.input_bit = DAC_precision_dict[DAC_choice]
        self.ADC_quantize_bit = ADC_precision_dict[ADC_choice]
        
        self.hardware_config['input_bit'] = self.input_bit
        self.hardware_config['ADC_quantize_bit'] = self.ADC_quantize_bit

        # group num
        self.pe_group_num = int(xbar_config.get('Process element level', 'Group_Num'))
        self.tile_size = list(map(int, xbar_config.get('Tile level', 'PE_Num').split(',')))
        self.tile_row = self.tile_size[0]
        self.tile_column = self.tile_size[1]
        
        # net and weights
        if device is None:
            self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(f'run on device {self.device}')

        if extra_define != None:
            self.hardware_config['input_bit'] = extra_define['dac_res']
            self.hardware_config['ADC_quantize_bit'] = extra_define['adc_res']
            self.hardware_config['xbar_size'] = extra_define['xbar_size']
        
        quantize_config_list, input_index_list = make_quantize_config_index_input_list(layer_config_list=layer_config_list)
        input_params = {'activation_scale': 1. / 255., 'activation_bit': 9, 'input_shape': input_shape}
        self.net = NetworkGraph(self.hardware_config, layer_config_list, quantize_config_list, input_index_list, input_params)

        if state_dict is not None:
            print(f'load state_dict')
            # load weights and split weights according to HW parameters
            #linqiushi modified
            self.net.load_change_weights(translate_state_dict_structure_file(state_dict=state_dict, quantize_config_list=quantize_config_list, structure_file=self.net.get_structure(), tmp=self.net.state_dict()))
            #linqiushi above

        if weights_file is not None and False:
            print(f'load weights from {weights_file}')
            # load weights and split weights according to HW parameters
            #linqiushi modified
            self.net.load_change_weights(torch.load(weights_file, map_location=self.device))
            # self.net.load_state_dict(torch.load(weights_file, map_location=self.device))
            #linqiushi above

def make_quantize_config_index_input_list(layer_config_list):
    quantize_config_list = []
    input_index_list = []
    for i in range(len(layer_config_list)):
        quantize_config_list.append({'weight_bit': 9, 'activation_bit':9, 'point_shift': -2})
        if 'input_index' in layer_config_list[i]:
            input_index_list.append(layer_config_list[i]['input_index'])
        else:
            # by default: the inputs of the current layer come from the outputs of the previous layer
            input_index_list.append([-1])
    return quantize_config_list, input_index_list