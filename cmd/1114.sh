python main.py --mode Train --model Multi_Res_tiny --exp_name Multi_Res_tiny
python main.py --mode Train --model Multi_Res_small --exp_name Multi_Res_small
# python main.py --mode Train --model Multi_Res_base --exp_name Multi_Res_base
python main.py --mode Train --model Multi_Res_large --exp_name Multi_Res_large


python main.py --mode Test --model Spectral_Res_SE_small --exp_name Spectral_Res_SE_small
python main.py --mode Test --model Multi_Res_small --exp_name Multi_Res_small
python main.py --mode Train --model Spectral_Res_SE_mini --exp_name Spectral_Res_SE_mini

python main.py --mode Train --model F_cnn_update --exp_name F_cnn_update



python main.py --mode Test --model rf_att_tiny --exp_name rf_att_tiny



####
python main.py --mode Test --model F_T_1114 --exp_name F_T_1114
python main.py --mode Train --model F_T_1114_low_rank --exp_name F_T_1114_low_rank
python main.py --mode Train --model F_T_1115_small2 --exp_name F_T_1115_small2
python main.py --mode Train --model F_T_1115_small2_normalization --exp_name F_T_1115_small2_normalization
python main.py --mode Train --model F_T_1115_small2_norm_multiloss --exp_name F_T_1115_small2_norm_multiloss
python main.py --mode Train --model F_T_1115_small2_multiloss --exp_name F_T_1115_small2_multiloss
python main.py --mode Train --model F_T_gate --exp_name F_T_gate



# A
python main.py --mode Train --model Fcnn_tiny --exp_name Fcnn_tiny

# B
python main.py --mode Train --model Tatt_tiny --exp_name Tatt_tiny
