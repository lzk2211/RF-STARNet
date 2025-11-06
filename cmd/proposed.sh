# 所提合并网络多信噪比训练
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-20dB/ --model ResInc_SpAtt --exp_name=-20dB_ResInc_SpAtt_1 --epochs 250 --lr 1e-4 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-15dB/ --model ResInc_SpAtt --exp_name=-15dB_ResInc_SpAtt_1 --epochs 250 --lr 1e-4 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-10dB/ --model ResInc_SpAtt --exp_name=-10dB_ResInc_SpAtt_1 --epochs 250 --lr 1e-4 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-5dB/ --model ResInc_SpAtt --exp_name=-5dB_ResInc_SpAtt_1 --epochs 250 --lr 1e-4 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_0dB/ --model ResInc_SpAtt --exp_name=0dB_ResInc_SpAtt_1 --epochs 250 --lr 1e-4 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_5dB/ --model ResInc_SpAtt --exp_name=5dB_ResInc_SpAtt_1 --epochs 250 --lr 1e-4 --batch_size 128 --class_num 10

# 37.5 56.3 74.7 86.6 93.3 94.1