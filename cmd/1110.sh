python main.py --mode Train --model rf_att_small --exp_name rf_att_small --epochs 160
python main.py --mode Train --model rf_att_medium --exp_name rf_att_medium --epochs 160
python main.py --mode Train --model rf_att_large --exp_name rf_att_large --epochs 160

python main.py --mode Train --model Res_Inc_Net_tiny --exp_name Res_Inc_Net_tiny
python main.py --mode Train --model Res_Inc_Net_small --exp_name Res_Inc_Net_small
python main.py --mode Train --model Res_Inc_Net_base --exp_name Res_Inc_Net_base
python main.py --mode Train --model Res_Inc_Net_large --exp_name Res_Inc_Net_large

python main.py --mode Test --model SW_Res_Inc_Net_base --exp_name SW_Res_Inc_Net_base
python main.py --mode Train --model SW_Res_Inc_Net_base_ECANet --exp_name SW_Res_Inc_Net_base_ECANet
python main.py --mode Train --model SW_Res_Inc_Net_base_SENet --exp_name SW_Res_Inc_Net_base_SENet

python main.py --mode Train --model rf_att_medium_Hybrid --exp_name rf_att_medium_Hybrid



python main.py --mode Train --model F_T --exp_name F_T
python main.py --mode Train --model F_T_news --exp_name F_T_news


python main.py --mode Train --model Res_Inc_Net_base_nodropout --exp_name Res_Inc_Net_base_nodropout --weight_decay 0.0


python main.py --mode Train --model F_T_finetune --exp_name F_T_finetune
python main.py --mode Train --model F_T_finetune2 --exp_name F_T_finetune2

python main.py --mode Train --model F_T_new_finetune --exp_name F_T_new_finetune


# 对照
python main.py --mode Test --model S3R --exp_name S3R

python main.py --mode Train --model MobileNetV3Small_ETF --exp_name MobileNetV3Small_ETF



python main.py --mode Train --model rf_att_medium_downsample --exp_name rf_att_medium_downsample
python main.py --mode Train --model rf_att_medium_downsample_16 --exp_name rf_att_medium_downsample_16


python main.py --mode Train --model HybridRFClassifier --exp_name HybridRFClassifier
