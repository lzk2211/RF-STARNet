# S3R多信噪比训练
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-20dB/ --model S3R --exp_name=-20dB_S3R_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-15dB/ --model S3R --exp_name=-15dB_S3R_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-10dB/ --model S3R --exp_name=-10dB_S3R_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-5dB/ --model S3R --exp_name=-5dB_S3R_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_0dB/ --model S3R --exp_name=0dB_S3R_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_5dB/ --model S3R --exp_name=5dB_S3R_1 --epochs 250 --lr 1e-4 --batch_size 48 --class_num 10 --device cuda


26.9 51.1 63.5 79.8 81.1 88.2
