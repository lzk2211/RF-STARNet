import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import os, gc
import json

import torchvision.models as models
from model.efficientNetb0 import EfficientNetB0
from model.LW_1D_resnet import *
from model.Res_Inc_Net import *
from model.MLP import *
from model.ResNet import *
from model.MIML_DCNN import MIML_DCNN
from model.Complex_ResNet import resnet34 as complex_resnet34, resnet18 as complex_resnet18
from model.Complex_ResNet_1D import resnet18 as complex_resnet18_1D, resnet34 as complex_resnet34_1D
from model.Complex_UNet import complex_unet
from model.TimeSeq_SA import TimeSeqSA
from model.LightAttentionNet import LightAttentionNet
from model.RFProjectedAttentionClassifier import RFProjectedAttentionClassifier
from model.RFFormer import RFFormer
from model.RFFormer_Gemini import RFFormer as RFFormer_Gemini
from model.RFFormer_Gemini2 import RFformer_v2 as RFFormer_Gemini2
from model.S3R import NET
from model.S3R_CNN import NET as S3R_CNN_NET
from model.S3R_SA import NET as S3R_SA_NET
from model.S3R_z import NET as S3R_z_NET
from model.ResInc_GAtt import ResInc_GAtt
from model.ResInc_with_S3R import NET as ResInc_with_S3R_NET
from model.ResInc_SpAtt import ResInc_SpAtt
from model.SpectraAtt_T import rf_projected_attention_medium as T_rf_projected_attention_medium
from model.SpectraAtt_F import rf_projected_attention_medium as F_rf_projected_attention_medium

from model.SpectraAtt import rf_projected_attention_large, rf_projected_attention_medium, rf_projected_attention_small
from model.SpectraAtt_noProjected import rf_projected_attention_large as No_projected_rf_projected_attention_large
from model.SpectraAtt_noProjected import rf_projected_attention_medium as No_projected_rf_projected_attention_medium



from model.ResInc_SpAtt_fusion import ResInc_SpAtt as ResInc_SpAtt_fusion

from model.MobileNetV3Small_ETF import MobileNetV3Small_ETF

from tabulate import tabulate

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def folder_settings(args, folder_path):
    release_gpu_memory()# 释放GPU内存

    # torch.manual_seed(42)

    # if you have GPU, the device will be cuda
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Different dataset have different operation
    # folder_name = os.path.basename(os.path.dirname(args.image_path))  # 提取D00
    # file_name = os.path.basename(args.image_path).replace("data_set_", "")  # 提取1024_1024
    # name = f"{folder_name}_{file_name}"

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers

    assert os.path.exists(args.image_path), "{} path does not exist.".format(args.image_path)

    # 创建表格数据
    table_data = [
        ["Device", device],
        ["Folder Path", folder_path],
        ["Batch Size", args.batch_size],
        ["Number of Workers", nw],
        ["Learning Rate", args.lr],
        ["Model", args.model],
        ["Class Number", args.class_num],
        ["Image Path", args.image_path]
    ]
    # 打印表格
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))
    return device, nw, table_data

def switch_model(model, device, num_classes):
    # Update the model to fit your input size and number of classes
    if model == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model == 'EfficientNetB0':
        model = EfficientNetB0(num_classes=num_classes)
    elif model == 'resnet34_vector_group':
        model = resnet34_vector_group(num_classes=num_classes)
    elif model == '1d_lw_resnet10_256':
        model = resnet10_vector_group(num_classes=num_classes)
    elif model == 'Res_Inc_Net':
        model = Res_Inc_Net(num_classes=num_classes)
    elif model == 'Res_Inc_Net_large':
        model = Res_Inc_Net_large(num_classes=num_classes)
    elif model == 'Res_Inc_Net_50':
        model = Res_Inc_Net_50(num_classes=num_classes, include_top=True)
    elif model == 'Res_Inc_Net_101':
        model = Res_Inc_Net_101(num_classes=num_classes, include_top=True)
    elif model == 'Res_Inc_Net_base':
        model = Res_Inc_Net_base(num_classes=num_classes, include_top=True)
    elif model == 'Res_Inc_Net_small':
        model = Res_Inc_Net_small(num_classes=num_classes, include_top=True)
    elif model == 'Res_Inc_Net_tiny':
        model = Res_Inc_Net_tiny(num_classes=num_classes, include_top=True)
    elif model == 'TimeSeqSA':
        model = TimeSeqSA(num_classes=num_classes)
    elif model == 'LightAttentionNet':
        model = LightAttentionNet(num_classes=num_classes, device=device, include_top=True)
    elif model == 'RFProjectedAttentionClassifier':
        model = RFProjectedAttentionClassifier(freq_bins=512, seq_len=128, dropout=0, num_classes=num_classes, groups=8, att_dim=32, k_len=64, device=device)

    elif model == 'rf_projected_attention_small':
        model = rf_projected_attention_small(freq_bins=512, seq_len=128, num_classes=num_classes)
    elif model == 'rf_projected_attention_medium':
        model = rf_projected_attention_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    elif model == 'rf_projected_attention_large':
        model = rf_projected_attention_large(freq_bins=512, seq_len=128, num_classes=num_classes)
    elif model == 'T_rf_projected_attention_medium':
        model = T_rf_projected_attention_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    elif model == 'F_rf_projected_attention_medium':
        model = F_rf_projected_attention_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    elif model == 'No_projected_rf_projected_attention_medium':
        model = No_projected_rf_projected_attention_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    elif model == 'No_projected_rf_projected_attention_large':
        model = No_projected_rf_projected_attention_large(freq_bins=512, seq_len=128, num_classes=num_classes)

    elif model == 'RFFormer':
        model = RFFormer(in_ch=1, num_classes=num_classes, embed_dim=160, depth=4, num_heads=5, patch_size=(4, 4))
    elif model == 'RFFormer_Gemini':
        model = RFFormer_Gemini(num_classes=num_classes, seq_len=128, embed_dim=128, num_layers=4, num_heads=4)
    elif model == 'RFFormer_Gemini2':
        model = RFFormer_Gemini2(num_classes=num_classes, patch_size_f=64, patch_size_t=8, embed_dim=128, num_layers=4, num_heads=4)
    elif model == 'S3R':
        model = NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    elif model == 'S3R_CNN':
        model = S3R_CNN_NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    elif model == 'S3R_SA':
        model = S3R_SA_NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    elif model == 'S3R_z':
        model = S3R_z_NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    elif model == 'ResInc_GAtt':
        model = ResInc_GAtt(num_classes=num_classes)
    elif model == 'ResInc_with_S3R':
        model = ResInc_with_S3R_NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    elif model == 'ResInc_SpAtt':
        model = ResInc_SpAtt(num_classes=num_classes)
    elif model == 'ResInc_SpAtt_fusion':
        model = ResInc_SpAtt_fusion(num_classes=num_classes, fusion_mode='bilinear')


    elif model == 'MobileNetV3Small_ETF':
        model = MobileNetV3Small_ETF(num_classes=num_classes)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)# two GPUs
    model.to(device)
    return model
