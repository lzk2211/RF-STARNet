#!/usr/bin/env bash
# 统一设置随机种子，支持外部覆盖：运行时可用 "SEED=123 ./1122.sh"
SEED=${SEED:-42}

# Fcnn 原实验
python main.py --mode Test --model F_cnn_bn_tiny_1122 --exp_name melting/F_cnn_bn_tiny_1122 --seed 42
python main.py --mode Test --model F_cnn_bn_small_1122 --exp_name melting/F_cnn_bn_small_1122 --seed 42

# 对照实验
python main.py --mode Test --model F_cnn_basic_tiny_1122 --exp_name melting/F_cnn_basic_tiny_1122 --seed 42
python main.py --mode Test --model F_cnn_MultiDilation_tiny_1122 --exp_name melting/F_cnn_MultiDilation_tiny_1122 --seed 42
python main.py --mode Test --model F_cnn_SE_tiny_1122 --exp_name melting/F_cnn_SE_tiny_1122 --seed 42
python main.py --mode Test --model F_cnn_basic_small_1122 --exp_name melting/F_cnn_basic_small_1122 --seed 42
python main.py --mode Test --model F_cnn_MultiDilation_small_1122 --exp_name melting/F_cnn_MultiDilation_small_1122 --seed 42
python main.py --mode Test --model F_cnn_SE_small_1122 --exp_name melting/F_cnn_SE_small_1122 --seed 42



# Tatt 原实验
python main.py --mode Test --model T_att_small_1122 --exp_name melting/T_att_small_1122 --seed 42
python main.py --mode Test --model T_att_tiny_1122 --exp_name melting/T_att_tiny_1122 --seed 42

# 对照实验
python main.py --mode Test --model T_att_tiny_nogroup_1122 --exp_name melting/T_att_tiny_nogroup_1122 --seed 42
python main.py --mode Train --model T_att_tiny_basic_1122 --exp_name melting/T_att_tiny_basic_1122 --seed 42 --batch_size 32
python main.py --mode Train --model T_att_tiny_nostem_1122 --exp_name melting/T_att_tiny_nostem_1122 --seed 42 --batch_size 32
python main.py --mode Test --model T_att_tiny_nobias_1122 --exp_name melting/T_att_tiny_nobias_1122 --seed 42
python main.py --mode Train --model T_att_tiny_basic_1122 --exp_name melting/T_att_tiny_basic_1122 --seed 42


python main.py --mode Test --model T_att_small_nogroup_1122 --exp_name melting/T_att_small_nogroup_1122 --seed 42
python main.py --mode Train --model T_att_small_basic_1122 --exp_name melting/T_att_small_basic_1122 --seed 42 --batch_size 32
python main.py --mode Train --model T_att_small_nostem_1122 --exp_name melting/T_att_small_nostem_1122 --seed 42 --batch_size 32
python main.py --mode Test --model T_att_small_nobias_1122 --exp_name melting/T_att_small_nobias_1122 --seed 42
python main.py --mode Train --model T_att_small_basic_1122 --exp_name melting/T_att_small_basic_1122 --seed 42



# F_T 原实验
python main.py --mode Test --model F_T_small_mulitloss_1122 --exp_name melting/F_T_small_mulitloss_1122 --seed 42
python main.py --mode Test --model F_T_tiny_mulitloss_1122 --exp_name melting/F_T_tiny_mulitloss_1122 --seed 42

# 对照实验
python main.py --mode Test --model F_T_small_basic_1122 --exp_name melting/F_T_small_basic_1122 --seed 42
python main.py --mode Test --model F_T_small_gate_1122 --exp_name melting/F_T_small_gate_1122 --seed 42
python main.py --mode Train --model F_T_small_mulitloss_butnogate_1122 --exp_name melting/F_T_small_mulitloss_butnogate_1122 --seed 42
python main.py --mode Test --model F_T_tiny_basic_1122 --exp_name melting/F_T_tiny_basic_1122 --seed 42
python main.py --mode Test --model F_T_tiny_gate_1122 --exp_name melting/F_T_tiny_gate_1122 --seed 42
python main.py --mode Train --model F_T_tiny_mulitloss_butnogate_1122 --exp_name melting/F_T_tiny_mulitloss_butnogate_1122 --seed 42












# python main.py --mode Train --model S3R --exp_name compare/S3R --seed 42