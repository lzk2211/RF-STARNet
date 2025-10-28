import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import os, gc
import json

import torchvision.models as models
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

from tabulate import tabulate

def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def folder_settings(args, folder_path):
    release_gpu_memory()# 释放GPU内存

    # torch.manual_seed(42)

    # if you have GPU, the device will be cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    elif model == 'resnet34_vector_group':
        model = resnet34_vector_group(num_classes=num_classes)
    elif model == '1d_lw_resnet10_256':
        model = resnet10_vector_group(num_classes=num_classes)
    elif model == 'Res_Inc_Net':
        model = Res_Inc_Net(num_classes=num_classes)
    elif model == 'TimeSeqSA':
        model = TimeSeqSA(num_classes=num_classes)
    elif model == 'LightAttentionNet':
        model = LightAttentionNet(num_classes=num_classes, device=device, include_top=True)
    elif model == 'RFProjectedAttentionClassifier':
        model = RFProjectedAttentionClassifier(freq_bins=128, seq_len=128, dropout=0, num_classes=num_classes, groups=8, att_dim=32, k_len=64, device=device)
    elif model == 'RFFormer':
        model = RFFormer(in_ch=1, num_classes=num_classes, embed_dim=160, depth=4, num_heads=5, patch_size=(4, 4))
    elif model == 'RFFormer_Gemini':
        model = RFFormer_Gemini(num_classes=num_classes, seq_len=128, embed_dim=128, num_layers=4, num_heads=4)
    elif model == 'RFFormer_Gemini2':
        model = RFFormer_Gemini2(num_classes=num_classes, patch_size_f=64, patch_size_t=8, embed_dim=128, num_layers=4, num_heads=4)
    elif model == 'S3R':
        model = NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    return model
