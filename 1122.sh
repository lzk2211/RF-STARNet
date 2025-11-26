#!/usr/bin/env bash
# 统一设置随机种子，支持外部覆盖：运行时可用 "SEED=123 ./1122.sh"
SEED=${SEED:-125}

python main.py --mode Test --model F_T_small_mulitloss_1122 --exp_name F_T_small_mulitloss_1122 --seed "$SEED"
python main.py --mode Test --model F_T_tiny_mulitloss_1122 --exp_name F_T_tiny_mulitloss_1122 --seed "$SEED"
python main.py --mode Test --model F_T_mini_mulitloss_1122 --exp_name F_T_mini_mulitloss_1122 --seed "$SEED"
python main.py --mode Test --model F_T_nano_mulitloss_1122 --exp_name F_T_nano_mulitloss_1122 --seed "$SEED"

python main.py --mode Test --model F_cnn_bn_nano_1122 --exp_name F_cnn_bn_nano_1122 --seed "$SEED"
python main.py --mode Test --model F_cnn_bn_mini_1122 --exp_name F_cnn_bn_mini_1122 --seed "$SEED"
python main.py --mode Test --model F_cnn_bn_tiny_1122 --exp_name F_cnn_bn_tiny_1122 --seed "$SEED"
python main.py --mode Test --model F_cnn_bn_small_1122 --exp_name F_cnn_bn_small_1122 --seed "$SEED"
python main.py --mode Test --model F_cnn_bn_base_1122 --exp_name F_cnn_bn_base_1122 --seed "$SEED"

python main.py --mode Test --model T_att_nano_1122 --exp_name T_att_nano_1122 --seed "$SEED"
python main.py --mode Test --model T_att_mini_1122 --exp_name T_att_mini_1122 --seed "$SEED"
python main.py --mode Test --model T_att_tiny_1122 --exp_name T_att_tiny_1122 --seed "$SEED"
python main.py --mode Test --model T_att_small_1122 --exp_name T_att_small_1122 --seed "$SEED"
python main.py --mode Test --model T_att_base_1122 --exp_name T_att_base_1122 --seed "$SEED"

