python main.py --Train True --Test True --class_num 10 --model 1d_lw_resnet10_256 --image_path '/media/lab125/SSD11/D00/data_set_5e6_-5dB/' --exp_name='-5dB_Res_Inc_Net_classification' --Dim 'f'

python main.py --Train True --Test True --class_num 10 --model LightAttentionNet --image_path '/media/lab125/SSD11/D00/data_set_5e6_-5dB/' --exp_name='-5dB_LightAttentionNet_classification' --Dim None

python main.py --Train True --Test True --class_num 10 --model RFProjectedAttentionClassifier --image_path '/media/lab125/SSD11/D00/data_set_5e6_-5dB/' --exp_name='-5dB_RFProjectedAttentionClassifier_F_classification' --Dim None


python main.py --Train True --Test True --class_num 10 --model resnet34 --image_path '/media/lab125/SSD11/D00/data_set_5e6_-5dB/' --exp_name='-5dB_Resnet_classification' --Dim None


python main.py --mode Train --model ResInc_SpAtt_noPro --exp_name ResInc_SpAtt_noPro --ratio 0.5
python main.py --mode Train --model F_T_gatefusion --exp_name F_T_gatefusion --ratio 0.5
python main.py --mode Train --model F_T_SENet --exp_name F_T_SENet --ratio 0.5
python main.py --mode Train --model F_T_sp_pool --exp_name F_T_sp_pool --ratio 0.5

