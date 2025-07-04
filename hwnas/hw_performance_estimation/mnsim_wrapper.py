import collections
import configparser
import math

import torch

from hw_performance_estimation.compability_layer import translate_state_dict_structure_file
from MNSIM.Interface.interface import TrainTestInterface
from MNSIM.Interface.network import NetworkGraph


class DatasetModuleDummy:
    def __init__(self, train_loader, val_loader):
        self.data_loaders = [train_loader, val_loader]

    def get_dataloader(self):
        return self.data_loaders

class WrappedTestTrainInterface(TrainTestInterface):
    """Copied from MNSIM-2.0 TrainTestInterface and more or less adapted."""

    def __init__(
        self,
        layer_config_list,
        network_module,
        train_loader,
        test_loader,
        sim_config_path,
        input_shape,
        state_dict,
        weights_file=None,
        device=None,
        extra_define=None,
    ):

        self.network_module = network_module  # Unused
        self.dataset_module = DatasetModuleDummy(train_loader, test_loader)
        self.weights_file = weights_file  # Unused
        self.test_loader = test_loader

        # load simconfig
        ## xbar_size, input_bit, weight_bit, ADC_quantize_bit
        xbar_config = configparser.ConfigParser()
        xbar_config.read(sim_config_path, encoding="UTF-8")
        self.hardware_config = collections.OrderedDict()

        # xbar_size
        xbar_size = list(
            map(int, xbar_config.get("Crossbar level", "Xbar_Size").split(","))
        )
        self.xbar_row = xbar_size[0]
        self.xbar_column = xbar_size[1]
        self.hardware_config["xbar_size"] = xbar_size[0]
        self.hardware_config["type"] = int(
            xbar_config.get("Process element level", "PIM_Type")
        )
        self.hardware_config["xbar_polarity"] = int(
            xbar_config.get("Process element level", "Xbar_Polarity")
        )
        self.hardware_config["DAC_num"] = int(
            xbar_config.get("Process element level", "DAC_Num")
        )

        # device bit
        self.device_bit = int(xbar_config.get("Device level", "Device_Level"))
        self.hardware_config["weight_bit"] = math.floor(math.log2(self.device_bit))

        # weight_bit means the weight bitwidth stored in one memory device
        # input bit and ADC bit
        ADC_choice = int(xbar_config.get("Interface level", "ADC_Choice"))
        DAC_choice = int(xbar_config.get("Interface level", "DAC_Choice"))
        temp_DAC_bit = int(xbar_config.get("Interface level", "DAC_Precision"))
        temp_ADC_bit = int(xbar_config.get("Interface level", "ADC_Precision"))
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
            9: 6,
        }
        DAC_precision_dict = {
            -1: temp_DAC_bit,
            1: 1,  # 1-bit
            2: 2,  # 2-bit
            3: 3,  # 3-bit
            4: 4,  # 4-bit
            5: 6,  # 6-bit
            6: 8,  # 8-bit
            7: 1,
        }
        self.input_bit = DAC_precision_dict[DAC_choice]
        self.ADC_quantize_bit = ADC_precision_dict[ADC_choice]

        self.hardware_config["input_bit"] = self.input_bit
        self.hardware_config["ADC_quantize_bit"] = self.ADC_quantize_bit

        # group num
        self.pe_group_num = int(xbar_config.get("Process element level", "Group_Num"))
        self.tile_size = list(
            map(int, xbar_config.get("Tile level", "PE_Num").split(","))
        )
        self.tile_row = self.tile_size[0]
        self.tile_column = self.tile_size[1]

        # net and weights
        if device is None:
            self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"run on device {self.device}")

        if extra_define != None:
            self.hardware_config["input_bit"] = extra_define["dac_res"]
            self.hardware_config["ADC_quantize_bit"] = extra_define["adc_res"]
            self.hardware_config["xbar_size"] = extra_define["xbar_size"]

        quantize_config_list, input_index_list = make_quantize_config_index_input_list(
            layer_config_list=layer_config_list
        )
        input_params = {
            "activation_scale": 1.0 / 255.0,
            "activation_bit": 9,
            "input_shape": input_shape,
        }
        self.net = NetworkGraph(
            self.hardware_config,
            layer_config_list,
            quantize_config_list,
            input_index_list,
            input_params,
        )

        if state_dict is not None:
            print(f"load state_dict")
            # load weights and split weights according to HW parameters
            # linqiushi modified
            self.net.load_change_weights(
                translate_state_dict_structure_file(
                    state_dict=state_dict,
                    quantize_config_list=quantize_config_list,
                    structure_file=self.net.get_structure(),
                    tmp=self.net.state_dict(),
                )
            )
            # linqiushi above

        if weights_file is not None and False:
            print(f"load weights from {weights_file}")
            # load weights and split weights according to HW parameters
            # linqiushi modified
            self.net.load_change_weights(
                torch.load(weights_file, map_location=self.device)
            )
            # self.net.load_state_dict(torch.load(weights_file, map_location=self.device))
            # linqiushi above


def make_quantize_config_index_input_list(layer_config_list):
    quantize_config_list = []
    input_index_list = []
    for i in range(len(layer_config_list)):
        quantize_config_list.append(
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2}
        )
        if "input_index" in layer_config_list[i]:
            input_index_list.append(layer_config_list[i]["input_index"])
        else:
            # by default: the inputs of the current layer come from the outputs of the previous layer
            input_index_list.append([-1])
    return quantize_config_list, input_index_list
