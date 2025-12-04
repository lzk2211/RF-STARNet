# A
python main.py --mode Test --model Fcnn_tiny --exp_name Fcnn_tiny
python main.py --mode Train --model Fcnn_noisegate_tiny --exp_name Fcnn_noisegate_tiny
python main.py --mode Train --model F_cnn_tiny --exp_name F_cnn_tiny
python main.py --mode Train --model F_cnn_mini --exp_name F_cnn_mini


# B
python main.py --mode Test --model Tatt_tiny --exp_name Tatt_tiny
python main.py --mode Train --model T_att_tiny --exp_name T_att_tiny
python main.py --mode Train --model T_att_mini --exp_name T_att_mini
python main.py --mode Train --model T_att_tiny_nozero --exp_name T_att_tiny_nozero
python main.py --mode Train --model T_att_medium_nozero --exp_name T_att_medium_nozero


# A+B
python main.py --mode Test --model F_T_tiny2 --exp_name F_T_tiny2
python main.py --mode Train --model F_T_mini2 --exp_name F_T_mini2
python main.py --mode Train --model F_T_mini2_1117 --exp_name F_T_mini2_1117
python main.py --mode Train --model F_T_mini2_1117_nogate --exp_name F_T_mini2_1117_nogate
python main.py --mode Train --model F_T_mini2_1117_gate --exp_name F_T_mini2_1117_gate

python main.py --mode Train --model F_T_mini2_gate_mulitloss --exp_name F_T_mini2_gate_mulitloss
python main.py --mode Train --model F_T_nano2_gate_mulitloss --exp_name F_T_nano2_gate_mulitloss
python main.py --mode Test --model F_T_mini2_gate_mulitloss_ortho --exp_name F_T_mini2_gate_mulitloss_ortho
python main.py --mode Train --model F_T_mini2_gate_mulitloss_ortho_DA --exp_name F_T_mini2_gate_mulitloss_ortho_DA
python main.py --mode Train --model F_T_mini2_gate_mulitloss_ortho_nozero --exp_name F_T_mini2_gate_mulitloss_ortho_nozero


python main.py --mode Test --model S3RImageNet_TinyTransformer --exp_name S3RImageNet_TinyTransformer
python main.py --mode Train --model S3R_1D_TinyTransformer --exp_name S3R_1D_TinyTransformer
python main.py --mode Train --model SpectralAtt --exp_name SpectralAtt_mini
python main.py --mode Train --model SpectralAtt2 --exp_name SpectralAtt2
python main.py --mode Train --model SpectralAtt2_base --exp_name SpectralAtt2_base

