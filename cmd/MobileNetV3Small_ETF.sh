# 轻量化论文网络多信噪比训练
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-20dB/ --model MobileNetV3Small_ETF --exp_name=-20dB_MobileNetV3Small_ETF_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda:0
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-15dB/ --model MobileNetV3Small_ETF --exp_name=-15dB_MobileNetV3Small_ETF_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda:0
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-10dB/ --model MobileNetV3Small_ETF --exp_name=-10dB_MobileNetV3Small_ETF_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda:0
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-5dB/ --model MobileNetV3Small_ETF --exp_name=-5dB_MobileNetV3Small_ETF_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda:0
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_0dB/ --model MobileNetV3Small_ETF --exp_name=0dB_MobileNetV3Small_ETF_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda:0
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_5dB/ --model MobileNetV3Small_ETF --exp_name=5dB_MobileNetV3Small_ETF_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda:0

/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6/ --model MobileNetV3Small_ETF --exp_name=MobileNetV3Small_ETF_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda:0

# 12.2 31.0 61.4 67.1 79.6 84.5 96.7

