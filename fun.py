import torch
import torch.nn as nn
import os, gc

from tabulate import tabulate

from model.Fcnn_Tatt_gate_mulitloss import F_T

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def folder_settings(args, folder_path):
    release_gpu_memory()

    # if you have GPU, the device will be cuda
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers

    assert os.path.exists(args.root_path), "{} path does not exist.".format(args.root_path)

    table_data = [
        ["Device", device],
        ["Folder Path", folder_path],
        ["Batch Size", args.batch_size],
        ["Number of Workers", nw],
        ["Learning Rate", args.lr],
        ["Model", args.model],
        ["Class Number", args.class_num],
        ["Image Path", args.root_path]
    ]
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))
    return device, nw, table_data

def switch_model(model, device, num_classes):
    # Update the model to fit your input size and number of classes
    if model == 'F_T_nano_mulitloss_1122':
        from model.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='nano')
    elif model == 'F_T_mini_mulitloss_1122':
        from model.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='mini')
    elif model == 'F_T_tiny_mulitloss_1122':
        from model.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='tiny')
    elif model == 'F_T_small_mulitloss_1122':  # small is the best one
        from model.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')

    elif model == 'F_cnn_bn_nano_1122':
        from model.F_cnn_bn import F_cnn_nano
        model = F_cnn_nano(num_classes=num_classes)
    elif model == 'F_cnn_bn_mini_1122':
        from model.F_cnn_bn import F_cnn_mini
        model = F_cnn_mini(num_classes=num_classes)
    elif model == 'F_cnn_bn_tiny_1122':
        from model.F_cnn_bn import F_cnn_tiny
        model = F_cnn_tiny(num_classes=num_classes)
    elif model == 'F_cnn_bn_small_1122': # small is the best one
        from model.F_cnn_bn import F_cnn_small
        model = F_cnn_small(num_classes=num_classes)
    elif model == 'F_cnn_bn_base_1122':
        from model.F_cnn_bn import F_cnn_base
        model = F_cnn_base(num_classes=num_classes)

    elif model == 'T_att_nano_1122':
        from model.T_att import T_att_nano
        model = T_att_nano(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_mini_1122':
        from model.T_att import T_att_mini
        model = T_att_mini(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_tiny_1122':
        from model.T_att import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_small_1122':  # small is the best one
        from model.T_att import T_att_small
        model = T_att_small(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_base_1122':
        from model.T_att import T_att_base
        model = T_att_base(freq_bins=512, seq_len=512, num_classes=num_classes)

    # comparsion
    elif model == 'F_cnn_basic_tiny_1122':
        from model.F_cnn_basic import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'F_cnn_SE_tiny_1122':
        from model.F_cnn_SE import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'F_cnn_MultiDilation_tiny_1122':
        from model.F_cnn_MultiDilation import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'F_cnn_basic_small_1122':
        from model.F_cnn_basic import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'F_cnn_SE_small_1122':
        from model.F_cnn_SE import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'F_cnn_MultiDilation_small_1122':
        from model.F_cnn_MultiDilation import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)

    elif model == 'T_att_tiny_nogroup_1122':
        from model.T_att_nogroup import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_small_nogroup_1122':
        from model.T_att_nogroup import T_att_small
        model = T_att_small(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_tiny_nostem_1122':
        from model.T_att_nostem import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_small_nostem_1122':
        from model.T_att_nostem import T_att_small
        model = T_att_small(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_tiny_nobias_1122':
        from model.T_att_nobias import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_small_nobias_1122':
        from model.T_att_nobias import T_att_small
        model = T_att_small(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_tiny_basic_1122':
        from model.T_att_basic import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_small_basic_1122':
        from model.T_att_basic import T_att_small
        model = T_att_small(freq_bins=512, seq_len=512, num_classes=num_classes)

    # 对照
    elif model == 'F_T_small_basic_1122':
        from model.Fcnn_Tatt import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')
    elif model == 'F_T_tiny_basic_1122':
        from model.Fcnn_Tatt import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='tiny')
    elif model == 'F_T_small_gate_1122':
        from model.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')
    elif model == 'F_T_tiny_gate_1122':
        from model.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='tiny')
    elif model == 'F_T_small_mulitloss_butnogate_1122':
        from model.Fcnn_Tatt_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')
    elif model == 'F_T_tiny_mulitloss_butnogate_1122':
        from model.Fcnn_Tatt_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='tiny')

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)# two GPUs
    model.to(device)
    return model
