# 只训练所提SA分支
# /home/lab125/anaconda3/envs/kk/bin/python main.py --model rf_projected_attention_large --exp_name=-5dB_rf_projected_attention_large_1 --epochs 250 --lr 1e-3 --batch_size 48 --class_num 10
# /home/lab125/anaconda3/envs/kk/bin/python main.py --model rf_projected_attention_large --exp_name=-5dB_rf_projected_attention_large_2 --epochs 250 --lr 1e-3 --batch_size 48 --class_num 10
# /home/lab125/anaconda3/envs/kk/bin/python main.py --model rf_projected_attention_large --exp_name=-5dB_rf_projected_attention_large_3 --epochs 250 --lr 1e-3 --batch_size 48 --class_num 10

# 83.9 + 84.0 + 84.4


/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-20dB/ --model rf_projected_attention_medium --exp_name=-20dB_rf_projected_attention_medium_1 --epochs 250 --lr 1e-3 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-15dB/ --model rf_projected_attention_medium --exp_name=-15dB_rf_projected_attention_medium_1 --epochs 250 --lr 1e-3 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-10dB/ --model rf_projected_attention_medium --exp_name=-10dB_rf_projected_attention_medium_1 --epochs 250 --lr 1e-3 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-5dB/ --model rf_projected_attention_medium --exp_name=-5dB_rf_projected_attention_medium_1 --epochs 250 --lr 1e-3 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_0dB/ --model rf_projected_attention_medium --exp_name=0dB_rf_projected_attention_medium_1 --epochs 250 --lr 1e-3 --batch_size 128 --class_num 10
/home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_5dB/ --model rf_projected_attention_medium --exp_name=5dB_rf_projected_attention_medium_1 --epochs 250 --lr 1e-3 --batch_size 128 --class_num 10

# 29.9 50.5 70.6 81.6 92.0 92.5




# /home/lab125/anaconda3/envs/kk/bin/python main.py --image_path /media/lab125/SSD11/D00/data_set_5e6_-5dB/ --model No_projected_rf_projected_attention_medium --exp_name=-5dB_No_projected_rf_projected_attention_medium_1 --epochs 250 --lr 1e-3 --batch_size 128 --class_num 10
