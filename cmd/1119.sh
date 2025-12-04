# # # HARNet
# python main.py --mode Test --model F_cnn_nano_1119 --exp_name F_cnn_nano_1119       --device cpu
# python main.py --mode Test --model F_cnn_mini_1119 --exp_name F_cnn_mini_1119       --device cpu
# python main.py --mode Test --model F_cnn_tiny_1119 --exp_name F_cnn_tiny_1119       --device cpu
# python main.py --mode Test --model F_cnn_small_1119 --exp_name F_cnn_small_1119     --device cpu
# python main.py --mode Test --model F_cnn_base_1119 --exp_name F_cnn_base_1119       --device cpu
# python main.py --mode Test --model F_cnn_large_1119 --exp_name F_cnn_large_1119     --device cpu


# # # SSTA
# python main.py --mode Test --model T_att_nano_1119 --exp_name T_att_nano_1119       --device cpu
# python main.py --mode Test --model T_att_mini_1119 --exp_name T_att_mini_1119       --device cpu
# python main.py --mode Test --model T_att_tiny_1119 --exp_name T_att_tiny_1119       --device cpu
# python main.py --mode Test --model T_att_small_1119 --exp_name T_att_small_1119     --device cpu
# python main.py --mode Test --model T_att_base_1119 --exp_name T_att_base_1119       --device cpu
# python main.py --mode Test --model T_att_large_1119 --exp_name T_att_large_1119     --device cpu


# # F_T
# python main.py --mode Test --model F_T_nano_gate_mulitloss_1119 --exp_name F_T_nano_gate_mulitloss_1119 --device cpu
# python main.py --mode Test --model F_T_mini_gate_mulitloss_1119 --exp_name F_T_mini_gate_mulitloss_1119 --device cpu
# python main.py --mode Test --model F_T_tiny_gate_mulitloss_1119 --exp_name F_T_tiny_gate_mulitloss_1119 --device cpu
# python main.py --mode Test --model F_T_small_gate_mulitloss_1119 --exp_name F_T_small_gate_mulitloss_1119 --device cpu
# python main.py --mode Test --model F_T_base_gate_mulitloss_1119 --exp_name F_T_base_gate_mulitloss_1119 --device cpu
# python main.py --mode Train --model F_T_large_gate_mulitloss_1119 --exp_name F_T_large_gate_mulitloss_1119

# F_对照
# python main.py --mode Train --model F_cnn_basic_tiny_1120 --exp_name F_cnn_basic_tiny_1120
# python main.py --mode Train --model F_cnn_basic_base_1120 --exp_name F_cnn_basic_base_1120
# python main.py --mode Train --model F_cnn_SE_tiny_1120 --exp_name F_cnn_SE_tiny_1120
# python main.py --mode Train --model F_cnn_SE_base_1120 --exp_name F_cnn_SE_base_1120
# python main.py --mode Train --model F_cnn_MultiDilation_tiny_1120 --exp_name F_cnn_MultiDilation_tiny_1120
# python main.py --mode Train --model F_cnn_MultiDilation_base_1120 --exp_name F_cnn_MultiDilation_base_1120

# T_对照
# python main.py --mode Train --model T_att_base_nogroup --exp_name T_att_base_nogroup
# python main.py --mode Train --model T_att_base_nostem --exp_name T_att_base_nostem
# python main.py --mode Train --model T_att_base_basic --exp_name T_att_base_basic
# python main.py --mode Train --model T_att_large_nogroup --exp_name T_att_large_nogroup
# python main.py --mode Train --model T_att_large_nostem --exp_name T_att_large_nostem
# python main.py --mode Train --model T_att_large_basic --exp_name T_att_large_basic

python main.py --mode Train --model F_T_small_basic --exp_name F_T_small_basic
python main.py --mode Train --model F_T_tiny_basic --exp_name F_T_tiny_basic
python main.py --mode Train --model F_T_small_gate --exp_name F_T_small_gate
python main.py --mode Train --model F_T_tiny_gate --exp_name F_T_tiny_gate

python main.py --mode Test --model resnet18_vector_group --exp_name resnet18_vector_group

python main.py --mode Train --model F_cnn_bn_small_1119 --exp_name F_cnn_bn_small_1119
