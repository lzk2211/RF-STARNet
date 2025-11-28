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
# from model.ResInc_GAtt import ResInc_GAtt
from model.ResInc_with_S3R import NET as ResInc_with_S3R_NET
from model.ResInc_SpAtt import ResInc_SpAtt
from model.ResInc_SpAtt_noProjected import ResInc_SpAtt as ResInc_SpAtt_noPro
# from model.SpectraAtt_T import rf_projected_attention_medium as T_rf_projected_attention_medium
# from model.SpectraAtt_F import rf_projected_attention_medium as F_rf_projected_attention_medium

from model.SpectraAtt import rf_projected_attention_large, rf_projected_attention_medium, rf_projected_attention_small
from model.SpectraAtt_noProjected import rf_projected_attention_large as No_projected_rf_projected_attention_large
from model.SpectraAtt_noProjected import rf_projected_attention_medium as No_projected_rf_projected_attention_medium



# from model.ResInc_SpAtt_fusion import ResInc_SpAtt as ResInc_SpAtt_fusion

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
    # folder_name = os.path.basename(os.path.dirname(args.root_path))  # 提取D00
    # file_name = os.path.basename(args.root_path).replace("data_set_", "")  # 提取1024_1024
    # name = f"{folder_name}_{file_name}"

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers

    assert os.path.exists(args.root_path), "{} path does not exist.".format(args.root_path)

    # 创建表格数据
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
    # 打印表格
    print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))
    return device, nw, table_data

def switch_model(model, device, num_classes):
    # Update the model to fit your input size and number of classes
    if model == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model == 'resnet18':
        model = resnet18(num_classes=num_classes)
    # elif model == 'EfficientNetB0':
    #     model = EfficientNetB0(num_classes=num_classes)
    # elif model == 'resnet34_vector_group':
    #     model = resnet34_vector_group(num_classes=num_classes)
    # elif model == '1d_lw_resnet10_256':
    #     model = resnet10_vector_group(num_classes=num_classes)
    # elif model == 'Res_Inc_Net':
    #     model = Res_Inc_Net(num_classes=num_classes)
    # elif model == 'Res_Inc_Net_large':
    #     model = Res_Inc_Net_large(num_classes=num_classes)
    # elif model == 'Res_Inc_Net_50':
    #     model = Res_Inc_Net_50(num_classes=num_classes, include_top=True)
    # elif model == 'Res_Inc_Net_101':
    #     model = Res_Inc_Net_101(num_classes=num_classes, include_top=True)
    # elif model == 'Res_Inc_Net_base':
    #     model = Res_Inc_Net_base(num_classes=num_classes, include_top=True)
    # elif model == 'Res_Inc_Net_small':
    #     model = Res_Inc_Net_small(num_classes=num_classes, include_top=True)
    # elif model == 'Res_Inc_Net_tiny':
    #     model = Res_Inc_Net_tiny(num_classes=num_classes, include_top=True)
    # elif model == 'TimeSeqSA':
    #     model = TimeSeqSA(num_classes=num_classes)
    # elif model == 'LightAttentionNet':
    #     model = LightAttentionNet(num_classes=num_classes, device=device, include_top=True)
    # elif model == 'RFProjectedAttentionClassifier':
    #     model = RFProjectedAttentionClassifier(freq_bins=512, seq_len=128, dropout=0, num_classes=num_classes, groups=8, att_dim=32, k_len=64, device=device)

    # elif model == 'rf_projected_attention_small':
    #     model = rf_projected_attention_small(freq_bins=512, seq_len=128, num_classes=num_classes)
    # elif model == 'rf_projected_attention_medium':
    #     model = rf_projected_attention_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    # elif model == 'rf_projected_attention_large':
    #     model = rf_projected_attention_large(freq_bins=512, seq_len=128, num_classes=num_classes)
    # elif model == 'T_rf_projected_attention_medium':
    #     model = T_rf_projected_attention_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    # elif model == 'F_rf_projected_attention_medium':
    #     model = F_rf_projected_attention_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    # elif model == 'No_projected_rf_projected_attention_medium':
    #     model = No_projected_rf_projected_attention_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    # elif model == 'No_projected_rf_projected_attention_large':
    #     model = No_projected_rf_projected_attention_large(freq_bins=512, seq_len=128, num_classes=num_classes)

    # elif model == 'RFFormer':
    #     model = RFFormer(in_ch=1, num_classes=num_classes, embed_dim=160, depth=4, num_heads=5, patch_size=(4, 4))
    # elif model == 'RFFormer_Gemini':
    #     model = RFFormer_Gemini(num_classes=num_classes, seq_len=128, embed_dim=128, num_layers=4, num_heads=4)
    # elif model == 'RFFormer_Gemini2':
    #     model = RFFormer_Gemini2(num_classes=num_classes, patch_size_f=64, patch_size_t=8, embed_dim=128, num_layers=4, num_heads=4)
    elif model == 'S3R':
        model = NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    # elif model == 'S3R_CNN':
    #     model = S3R_CNN_NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    # elif model == 'S3R_SA':
    #     model = S3R_SA_NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    # elif model == 'S3R_z':
    #     model = S3R_z_NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    # elif model == 'ResInc_GAtt':
    #     model = ResInc_GAtt(num_classes=num_classes)
    # elif model == 'ResInc_with_S3R':
    #     model = ResInc_with_S3R_NET(in_channels=1, input_size=[int(512), 512], semantic_dim=128, num_class=num_classes, device=device)
    # elif model == 'ResInc_SpAtt':
    #     model = ResInc_SpAtt(num_classes=num_classes)
    # elif model == 'ResInc_SpAtt_noPro':
    #     model = ResInc_SpAtt_noPro(num_classes=num_classes)


    elif model == 'MobileNetV3Small_ETF':
        model = MobileNetV3Small_ETF(num_classes=num_classes)
    elif model == 'IM_MDSA':
        from model1113.im_mdsa_model import IM_MDSA
        model = IM_MDSA(num_classes=num_classes)


    # elif model == 'F_T_gatefusion':
    #     from model.Fcnn_Tatt_gatefusion import ResInc_SpAtt as F_T_gatefusion
    #     model = F_T_gatefusion(num_classes=num_classes)
    # elif model == 'F_T_SENet':
    #     from model.Fcnn_Tatt_SENet import ResInc_SpAtt as F_T_SENet
    #     model = F_T_SENet(num_classes=num_classes)
    # elif model == 'F_T_sp_pool':
    #     from model.Fcnn_Tatt_sp_pool import ResInc_SpAtt as F_T_sp_pool
    #     model = F_T_sp_pool(num_classes=num_classes)



    # elif model == 'F_T_news':
    #     from newmodels.Fcnn_Tatt import ResInc_SpAtt as F_T_news
    #     model = F_T_news(num_classes=10)


# 1110
    elif model == 'Res_Inc_Net_tiny':
        from newmodels.Res_Inc_Net import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes, include_top=True)
    elif model == 'Res_Inc_Net_small':
        from newmodels.Res_Inc_Net import Res_Inc_Net_small
        model = Res_Inc_Net_small(num_classes=num_classes, include_top=True)
    elif model == 'Res_Inc_Net_base':
        from newmodels.Res_Inc_Net import Res_Inc_Net_base
        model = Res_Inc_Net_base(num_classes=num_classes, include_top=True)
    elif model == 'Res_Inc_Net_base_nodropout':
        from newmodels.Res_Inc_Net import Res_Inc_Net_base
        model = Res_Inc_Net_base(num_classes=num_classes, include_top=True, dropout=0.0)
    elif model == 'Res_Inc_Net_base_bigdropout':
        from newmodels.Res_Inc_Net import Res_Inc_Net_base
        model = Res_Inc_Net_base(num_classes=num_classes, include_top=True, dropout=0.75)

    elif model == 'Res_Inc_Net_large':
        from newmodels.Res_Inc_Net import Res_Inc_Net_large
        model = Res_Inc_Net_large(num_classes=num_classes, include_top=True)

    elif model == 'Multi_Res_tiny':
        from model1113.Multidilation_ResNet import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes, include_top=True)
    elif model == 'Multi_Res_small':
        from model1113.Multidilation_ResNet import Res_Inc_Net_small
        model = Res_Inc_Net_small(num_classes=num_classes, include_top=True)
    elif model == 'Multi_Res_base':
        from model1113.Multidilation_ResNet import Res_Inc_Net_base
        model = Res_Inc_Net_base(num_classes=num_classes, include_top=True)
    elif model == 'Multi_Res_large':
        from model1113.Multidilation_ResNet import Res_Inc_Net_large
        model = Res_Inc_Net_large(num_classes=num_classes, include_top=True)
    elif model == 'Spectral_Res_SE_small':
        from model1113.Spectral_Resolution_old import Res_Inc_Net_small
        model = Res_Inc_Net_small(num_classes=num_classes, include_top=True)


    elif model == 'SW_Res_Inc_Net_base':
        from newmodels.Res_Inc_Net_SWConv import SW_Res_Inc_Net_base
        model = SW_Res_Inc_Net_base(num_classes=num_classes, include_top=True)
    elif model == 'SW_Res_Inc_Net_base_SENet':
        from newmodels.Res_Inc_Net_SWConv_ECANet import SW_Res_Inc_Net_base as SW_Res_Inc_Net_base_SENet
        model = SW_Res_Inc_Net_base_SENet(num_classes=num_classes, include_top=True)
    elif model == 'SW_Res_Inc_Net_small_SENet':
        from model1113.Res_Inc_Net_SWConv_SENet import SW_Res_Inc_Net_small as SW_Res_Inc_Net_small_SENet
        model = SW_Res_Inc_Net_small_SENet(num_classes=num_classes, include_top=True)
    elif model == 'SW_Res_Inc_Net_small_FCDA':
        from model1113.Res_Inc_Net_SWConv_FCDA import SW_Res_Inc_Net_small as SW_Res_Inc_Net_small_FCDA
        model = SW_Res_Inc_Net_small_FCDA(num_classes=num_classes, include_top=True)


    elif model == 'SW_Res_Inc_Net_base_ECANet':
        from newmodels.Res_Inc_Net_SWConv_ECANet import SW_Res_Inc_Net_base as SW_Res_Inc_Net_base_ECANet
        model = SW_Res_Inc_Net_base_ECANet(num_classes=num_classes, include_top=True)

    elif model == 'rf_att_small':
        from newmodels.SpectraAtt_noProjected import rf_att_small
        model = rf_att_small(freq_bins=512, seq_len=128, num_classes=num_classes)
    elif model == 'rf_att_medium':
        from newmodels.SpectraAtt_noProjected import rf_att_medium
        model = rf_att_medium(freq_bins=512, seq_len=128, num_classes=num_classes)
    elif model == 'rf_att_medium_downsample':
        from newmodels.SpectraAtt_noProjected_downsample import rf_att_medium
        model = rf_att_medium(freq_bins=512, seq_len=128, num_classes=num_classes, stem_pool_factor=8)
    elif model == 'rf_att_medium_downsample_16':
        from newmodels.SpectraAtt_noProjected_downsample import rf_att_medium
        model = rf_att_medium(freq_bins=512, seq_len=128, num_classes=num_classes, stem_pool_factor=16)
    elif model == 'rf_att_tiny_downsample_32':
        from newmodels.SpectraAtt_noProjected_downsample import rf_att_tiny
        model = rf_att_tiny(freq_bins=512, seq_len=128, num_classes=num_classes)


    elif model == 'rf_att_large':
        from newmodels.SpectraAtt_noProjected import rf_att_large
        model = rf_att_large(freq_bins=512, seq_len=128, num_classes=num_classes)

    elif model == 'rf_att_medium_Hybrid':
        from newmodels.SpectraAtt_Hybrid import rf_att_medium as rf_att_medium_Hybrid
        model = rf_att_medium_Hybrid(freq_bins=512, seq_len=128, num_classes=num_classes)



    elif model == 'F_T':
        from newmodels.Fcnn_Tatt import F_T
        model = F_T(num_classes=num_classes)
    elif model == 'rf_att_tiny':
        from newmodels.T_att import rf_att_tiny
        model = rf_att_tiny(freq_bins=512, seq_len=512, num_classes=10, include_top=True)


# 1113
    elif model == 'HAR_Net_tiny':
        from model1113.HAR import HAR_Net_tiny
        model = HAR_Net_tiny(num_classes=num_classes)
    elif model == 'HAR_Net_small':
        from model1113.HAR import HAR_Net_small
        model = HAR_Net_small(num_classes=num_classes)
    elif model == 'HAR_Net_base':
        from model1113.HAR import HAR_Net_base
        model = HAR_Net_base(num_classes=num_classes)

    elif model == 'SpectralResolution_tiny':
        from model1113.Spectral_Resolution_old import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)

    elif model == 'SpectralResolution_small':
        from model1113.Spectral_Resolution_old import Res_Inc_Net_small
        model = Res_Inc_Net_small(num_classes=num_classes)

    elif model == 'SpectralResolution_small_2_SE':
        from model1113.Spectral_Resolution2_SE import Res_Inc_Net_small
        model = Res_Inc_Net_small(num_classes=num_classes)

    elif model == 'SpectralResolutionSE_small':
        from model1113.Spectral_Resolution2_SE import Res_Inc_Net_small
        model = Res_Inc_Net_small(num_classes=num_classes)


# 1114
    elif model == 'F_T_1114':
        from model1113.Fcnn_Tatt import F_T
        model = F_T(num_classes=num_classes)
    elif model == 'F_T_1114_low_rank':
        from model1115.Fcnn_Tatt import F_T
        model = F_T(num_classes=num_classes)
    elif model == 'F_T_1115_small2':
        from model1115.Fcnn_Tatt import ResInc_SpAtt_Classifier
        model = ResInc_SpAtt_Classifier(num_classes=num_classes)
    elif model == 'F_T_1115_small2_multiloss':
        from model1115.Fcnn_Tatt import ResInc_SpAtt_Classifier
        model = ResInc_SpAtt_Classifier(num_classes=num_classes)
    elif model == 'F_cnn_update':
        from model1115.Spectral_Resolution_update import Res_Inc_Net_small
        model = Res_Inc_Net_small(num_classes=num_classes)
    elif model == 'F_T_gate':
        from model1115.Fcnn_Tatt_gate import ResInc_SpAtt_Classifier
        model = ResInc_SpAtt_Classifier(num_classes=num_classes)


# 1115
    elif model == 'Fcnn_tiny':
        from model1115.Spectral_Resolution import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'Tatt_tiny':
        from model1115.T_att import rf_att_tiny
        model = rf_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'F_T_tiny2':
        from model1117.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes)
    elif model == 'F_T_mini2':#其实是tiny
        from model1115.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes)

    elif model == 'F_T_mini2_1117':
        from model1117.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes)
    elif model == 'F_T_mini2_1117_nogate':
        from model1117.Fcnn_Tatt import F_T
        model = F_T(num_classes=num_classes)
    elif model == 'F_T_mini2_1117_gate':
        from model1117.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes)
    elif model == 'F_T_mini2_1117_gate':
        from model1117.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes)
    elif model == 'T_att_tiny':
        from model1117.T_att import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_tiny_nozero':
        from model1117.T_att import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_medium_nozero':
        from model1117.T_att import T_att_medium
        model = T_att_medium(freq_bins=512, seq_len=512, num_classes=num_classes)

    elif model == 'F_cnn_tiny':
        from model1117.F_cnn import F_cnn_tiny
        model = F_cnn_tiny(num_classes=num_classes)
    elif model == 'T_att_mini':
        from model1117.T_att import T_att_mini
        model = T_att_mini(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'F_cnn_mini':
        from model1117.F_cnn import F_cnn_mini
        model = F_cnn_mini(num_classes=num_classes)

    elif model == 'F_T_mini2_gate_mulitloss':
        from model1117.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='mini')
    elif model == 'F_T_mini2_gate_mulitloss_ortho':
        from model1117.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='mini')
    elif model == 'F_T_mini2_gate_mulitloss_ortho_nozero':
        from model1117.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='mini')
    elif model == 'F_T_mini2_gate_mulitloss_ortho_DA':
        from model1117.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='mini')
    elif model == 'F_T_nano2_gate_mulitloss':
        from model1117.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='nano')

    elif model == 'S3RImageNet_TinyTransformer':
        from model1115.yll import S3RImageNet_TinyTransformer
        model = S3RImageNet_TinyTransformer(num_classes=num_classes)
    elif model == 'Fcnn_noisegate_tiny':
        from model1115.Spectral_Resolution_noise_gate import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'S3R_1D_TinyTransformer':
        from model1115.S3R_light import S3R_1D_TinyTransformer
        model = S3R_1D_TinyTransformer(num_classes=num_classes)

    elif model == 'SpectralAtt':
        from model1115.Spectral_selfatt import SpectralAttentionNet
        model = SpectralAttentionNet(num_classes=num_classes)
    elif model == 'SpectralAtt2':
        from model1115.Spectral_selfatt2 import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'SpectralAtt2_base':
        from model1115.Spectral_selfatt2 import Res_Inc_Net_base
        model = Res_Inc_Net_base(num_classes=num_classes)

# 1119
    elif model == 'F_cnn_nano_1119':
        from model1119.F_cnn import F_cnn_nano
        model = F_cnn_nano(num_classes=num_classes)
    elif model == 'F_cnn_mini_1119':
        from model1119.F_cnn import F_cnn_mini
        model = F_cnn_mini(num_classes=num_classes)
    elif model == 'F_cnn_tiny_1119':
        from model1119.F_cnn import F_cnn_tiny
        model = F_cnn_tiny(num_classes=num_classes)
    elif model == 'F_cnn_small_1119':
        from model1119.F_cnn import F_cnn_small
        model = F_cnn_small(num_classes=num_classes)
    elif model == 'F_cnn_base_1119':
        from model1119.F_cnn import F_cnn_base
        model = F_cnn_base(num_classes=num_classes)
    elif model == 'F_cnn_large_1119':
        from model1119.F_cnn import F_cnn_large
        model = F_cnn_large(num_classes=num_classes)

    ###############################
    elif model == 'T_att_nano_1119':
        from model1119.T_att import T_att_nano
        model = T_att_nano(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_mini_1119':
        from model1119.T_att import T_att_mini
        model = T_att_mini(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_tiny_1119':
        from model1119.T_att import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_small_1119':
        from model1119.T_att import T_att_small
        model = T_att_small(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_base_1119':
        from model1119.T_att import T_att_base
        model = T_att_base(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_large_1119':
        from model1119.T_att import T_att_large
        model = T_att_large(freq_bins=512, seq_len=512, num_classes=num_classes)
    ###############################
    elif model == 'F_T_nano_gate_mulitloss_1119':
        from model1119.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='nano')
    elif model == 'F_T_mini_gate_mulitloss_1119':
        from model1119.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='mini')
    elif model == 'F_T_tiny_gate_mulitloss_1119':
        from model1119.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='tiny')
    elif model == 'F_T_small_gate_mulitloss_1119':
        from model1119.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')
    elif model == 'F_T_base_gate_mulitloss_1119':
        from model1119.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='base')
    elif model == 'F_T_large_gate_mulitloss_1119':
        from model1119.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='large')
    ###############################


    # 对照
    elif model == 'F_cnn_basic_tiny_1120':
        from model1119.F_cnn_basic import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'F_cnn_basic_base_1120':
        from model1119.F_cnn_basic import Res_Inc_Net_base
        model = Res_Inc_Net_base(num_classes=num_classes)
    elif model == 'F_cnn_SE_tiny_1122':
        from model1122.F_cnn_SE import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'F_cnn_SE_base_1122':
        from model1122.F_cnn_SE import Res_Inc_Net_base
        model = Res_Inc_Net_base(num_classes=num_classes)
    elif model == 'F_cnn_MultiDilation_tiny_1120':
        from model1119.F_cnn_MultiDilation import Res_Inc_Net_tiny
        model = Res_Inc_Net_tiny(num_classes=num_classes)
    elif model == 'F_cnn_MultiDilation_base_1120':
        from model1119.F_cnn_MultiDilation import Res_Inc_Net_base
        model = Res_Inc_Net_base(num_classes=num_classes)
    # 对照
    elif model == 'T_att_large_nogroup':
        from model1119.T_att_nogroup import T_att_large
        model = T_att_large(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_base_nogroup':
        from model1119.T_att_nogroup import T_att_base
        model = T_att_base(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_large_nostem':
        from model1119.T_att_nostem import T_att_large
        model = T_att_large(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_base_nostem':
        from model1119.T_att_nostem import T_att_base
        model = T_att_base(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_large_basic':
        from model1119.T_att_basic import T_att_large
        model = T_att_large(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_base_basic':
        from model1119.T_att_basic import T_att_base
        model = T_att_base(freq_bins=512, seq_len=512, num_classes=num_classes)
    # 对照
    elif model == 'F_T_small_basic':
        from model1119.Fcnn_Tatt import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')
    elif model == 'F_T_tiny_basic':
        from model1119.Fcnn_Tatt import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='tiny')
    elif model == 'F_T_small_gate':
        from model1119.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')
    elif model == 'F_T_tiny_gate':
        from model1119.Fcnn_Tatt_gate import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='tiny')


    elif model == 'ASA_RFFI_Net':
        from model1119.ASA_RFFI_Model import ASA_RFFI_Net
        model = ASA_RFFI_Net(num_classes=num_classes)
    elif model == 'resnet18_vector_group':
        from model1119.ResNet_vector_group import resnet18_vector_group
        model = resnet18_vector_group(num_classes=num_classes)

# 1122
    elif model == 'F_T_small_mulitloss_1122':
        from model1122.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')
    elif model == 'F_T_tiny_mulitloss_1122':
        from model1122.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='tiny')
    elif model == 'F_T_mini_mulitloss_1122':
        from model1122.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='mini')
    elif model == 'F_T_nano_mulitloss_1122':
        from model1122.Fcnn_Tatt_gate_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='nano')

    elif model == 'F_cnn_bn_nano_1122':
        from model1122.F_cnn_bn import F_cnn_nano
        model = F_cnn_nano(num_classes=num_classes)
    elif model == 'F_cnn_bn_mini_1122':
        from model1122.F_cnn_bn import F_cnn_mini
        model = F_cnn_mini(num_classes=num_classes)
    elif model == 'F_cnn_bn_tiny_1122':
        from model1122.F_cnn_bn import F_cnn_tiny
        model = F_cnn_tiny(num_classes=num_classes)
    elif model == 'F_cnn_bn_small_1122':
        from model1122.F_cnn_bn import F_cnn_small
        model = F_cnn_small(num_classes=num_classes)
    elif model == 'F_cnn_bn_base_1122':
        from model1122.F_cnn_bn import F_cnn_base
        model = F_cnn_base(num_classes=num_classes)

    elif model == 'T_att_nano_1122':
        from model1122.T_att import T_att_nano
        model = T_att_nano(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_mini_1122':
        from model1122.T_att import T_att_mini
        model = T_att_mini(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_tiny_1122':
        from model1122.T_att import T_att_tiny
        model = T_att_tiny(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_small_1122':
        from model1122.T_att import T_att_small
        model = T_att_small(freq_bins=512, seq_len=512, num_classes=num_classes)
    elif model == 'T_att_base_1122':
        from model1122.T_att import T_att_base
        model = T_att_base(freq_bins=512, seq_len=512, num_classes=num_classes)

    elif model == 'F_T_small_mulitloss_1127':
        from model1122.Fcnn_Tatt_mulitloss import F_T
        model = F_T(num_classes=num_classes, use_resinc_variant='small')


    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)# two GPUs
    model.to(device)
    return model
