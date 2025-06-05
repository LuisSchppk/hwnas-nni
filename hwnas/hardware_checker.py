import time
from pytz import utc
from sympy import true
import MNSIM
from util import trace_model
from MNSIM.Accuracy_Model.Weight_update import weight_update
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Energy_Model import Model_energy
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Power_Model import Model_inference_power
from mnsim_wrapper import WrappedTestTrainInterface

def test(model, train_loader, val_loader):

    input_shape = (1, 1, 128, 128)
    layer_config_list = trace_model(model=model, input_shape=input_shape, num_classes=100)
    num_classes = 100
    
    # input_shape = (1, 3, 32, 32)
    # layer_config_list = []
    # layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 6, 'kernel_size': 5})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
    # layer_config_list.append({'type': 'conv', 'in_channels': 6, 'out_channels': 16, 'kernel_size': 5})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
    # layer_config_list.append({'type': 'conv', 'in_channels': 16, 'out_channels': 120, 'kernel_size': 5})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'view'})
    # layer_config_list.append({'type': 'fc', 'in_features': 120, 'out_features': 84})
    # layer_config_list.append({'type': 'dropout'})
    # layer_config_list.append({'type': 'relu'})
    # layer_config_list.append({'type': 'fc', 'in_features': 84, 'out_features': num_classes})

    network_module = None # not used in this version
    state_dict = model.state_dict()
    train_loader = train_loader
    val_loader = val_loader
    sim_config_path = "MNSIM-2.0/SimConfig.ini"
    enable_variation = True
    enable_SAF = True
    enable_R_ratio = True
    enable_fixed_Qrange = True

    print(model)
    print(layer_config_list)
    test_train_interface = WrappedTestTrainInterface(layer_config_list=layer_config_list, network_module=network_module, train_loader=train_loader, val_loader=val_loader, sim_config_path=sim_config_path, input_shape=input_shape, state_dict=state_dict)

    structure_file = test_train_interface.get_structure()
    TCG_mapping = TCG(structure_file, sim_config_path)

    hardware_modeling_start_time = time.time()
    latency = Model_latency(NetStruct=structure_file, SimConfig_path=sim_config_path, TCG_mapping=TCG_mapping)
    latency.calculate_model_latency(mode=1)
    print("========================Latency Results=================================")
    latency.model_latency_output()

    area = Model_area(NetStruct=structure_file, SimConfig_path=sim_config_path, TCG_mapping=TCG_mapping)
    print("========================Area Results=================================")
    area.model_area_output()

    power = Model_inference_power(NetStruct=structure_file, SimConfig_path=sim_config_path, TCG_mapping=TCG_mapping)
    print("========================Power Results=================================")
    power.model_power_output()

    energy = Model_energy(NetStruct=structure_file, SimConfig_path=sim_config_path,
                                TCG_mapping=TCG_mapping, model_latency=latency, model_power=power)
    print("========================Energy Results=================================")
    energy.model_energy_output()
    hardware_modeling_end_time = time.time()

    print("======================================")
    print("Accuracy simulation will take a few minutes on GPU")
    accuracy_modeling_start_time = time.time()
    weight = test_train_interface.get_net_bits()
    weight_2 = weight_update(sim_config_path, weight, is_Variation=enable_variation, is_SAF=enable_SAF, is_Rratio=enable_R_ratio)

    if not (enable_fixed_Qrange):
        print("Original accuracy:", test_train_interface.origin_evaluate(method='FIX_TRAIN', adc_action='SCALE'))
        print("PIM-based computing accuracy:", test_train_interface.set_net_bits_evaluate(weight_2, adc_action='SCALE'))
    else:
        print("Original accuracy:", test_train_interface.origin_evaluate(method='FIX_TRAIN', adc_action='FIX'))
        print("PIM-based computing accuracy:", test_train_interface.set_net_bits_evaluate(weight_2, adc_action='FIX'))
    accuracy_modeling_end_time = time.time()

    hardware_modeling_time = hardware_modeling_end_time - hardware_modeling_start_time
    accuracy_modeling_time = accuracy_modeling_end_time - accuracy_modeling_start_time
    print("Hardware modeling time:", hardware_modeling_time)
    print("Accuracy modeling time:", accuracy_modeling_time)