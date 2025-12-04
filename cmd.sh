#!/usr/bin/env bash
# 统一设置随机种子，支持外部覆盖：运行时可用 "SEED=123 ./1122.sh"
SEED=${SEED:-42}

python main.py --mode Train --model F_T_small_mulitloss_1122 --exp_name F_T_small_mulitloss_1122 --seed "$SEED"
python main.py --mode Train --model F_T_tiny_mulitloss_1122 --exp_name F_T_tiny_mulitloss_1122 --seed "$SEED"
python main.py --mode Train --model F_T_mini_mulitloss_1122 --exp_name F_T_mini_mulitloss_1122 --seed "$SEED"
python main.py --mode Train --model F_T_nano_mulitloss_1122 --exp_name F_T_nano_mulitloss_1122 --seed "$SEED"

python main.py --mode Train --model F_cnn_bn_nano_1122 --exp_name F_cnn_bn_nano_1122 --seed "$SEED"
python main.py --mode Train --model F_cnn_bn_mini_1122 --exp_name F_cnn_bn_mini_1122 --seed "$SEED"
python main.py --mode Train --model F_cnn_bn_tiny_1122 --exp_name F_cnn_bn_tiny_1122 --seed "$SEED"
python main.py --mode Train --model F_cnn_bn_small_1122 --exp_name F_cnn_bn_small_1122 --seed "$SEED"
python main.py --mode Train --model F_cnn_bn_base_1122 --exp_name F_cnn_bn_base_1122 --seed "$SEED"

python main.py --mode Train --model T_att_nano_1122 --exp_name T_att_nano_1122 --seed "$SEED"
python main.py --mode Train --model T_att_mini_1122 --exp_name T_att_mini_1122 --seed "$SEED"
python main.py --mode Train --model T_att_tiny_1122 --exp_name T_att_tiny_1122 --seed "$SEED"
python main.py --mode Train --model T_att_small_1122 --exp_name T_att_small_1122 --seed "$SEED"
python main.py --mode Train --model T_att_base_1122 --exp_name T_att_base_1122 --seed "$SEED"




# python main.py --mode Test --model F_cnn_bn_tiny_1122 --exp_name melting/F_cnn_bn_tiny_1122 --seed 42

# 对照实验
python main.py --mode Test --model F_cnn_basic_tiny_1122 --exp_name melting/F_cnn_basic_tiny_1122 --seed 42
python main.py --mode Test --model F_cnn_MultiDilation_tiny_1122 --exp_name melting/F_cnn_MultiDilation_tiny_1122 --seed 42
python main.py --mode Test --model F_cnn_SE_tiny_1122 --exp_name melting/F_cnn_SE_tiny_1122 --seed 42
python main.py --mode Train --model F_cnn_basic_small_1122 --exp_name melting/F_cnn_basic_small_1122 --seed 42
python main.py --mode Train --model F_cnn_MultiDilation_small_1122 --exp_name melting/F_cnn_MultiDilation_small_1122 --seed 42
python main.py --mode Train --model F_cnn_SE_small_1122 --exp_name melting/F_cnn_SE_small_1122 --seed 42


# 对照实验
python main.py --mode Train --model T_att_tiny_nogroup_1122 --exp_name melting/T_att_tiny_nogroup_1122 --seed 42
python main.py --mode Train --model T_att_small_nogroup_1122 --exp_name melting/T_att_small_nogroup_1122 --seed 42
python main.py --mode Train --model T_att_tiny_nostem_1122 --exp_name melting/T_att_tiny_nostem_1122 --seed 42
python main.py --mode Train --model T_att_small_nostem_1122 --exp_name melting/T_att_small_nostem_1122 --seed 42


python main.py --mode Train --model F_T_small_basic_1122 --exp_name melting/F_T_small_basic_1122 --seed 42
python main.py --mode Train --model F_T_tiny_basic_1122 --exp_name melting/F_T_tiny_basic_1122 --seed 42
python main.py --mode Train --model F_T_small_gate_1122 --exp_name melting/F_T_small_gate_1122 --seed 42
python main.py --mode Train --model F_T_tiny_gate_1122 --exp_name melting/F_T_tiny_gate_1122 --seed 42
python main.py --mode Train --model F_T_small_multiloss_1122 --exp_name melting/F_T_small_multiloss_1122 --seed 42
python main.py --mode Train --model F_T_tiny_multiloss_1122 --exp_name melting/F_T_tiny_multiloss_1122 --seed 42

