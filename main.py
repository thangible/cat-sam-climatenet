import os
import time
os.system("python train.py --config input_config --shot_num 16 --wandb --cat_type cat-a --run_name 16_a")
time.sleep(20)
os.system("python train.py --config input_config --shot_num 16 --wandb --cat_type cat-t --run_name 16_t")
time.sleep(20)
os.system("python train.py --config input_config --shot_num 1 --wandb --cat_type cat-t --run_name 1_t")
time.sleep(20)
os.system("python train.py --config input_config  --wandb --cat_type cat-t --run_name full_t")
time.sleep(20)
os.system("python train.py --config input_config  --wandb --cat_type cat-a --run_name full_a")
time.sleep(20)
os.system("python train.py --config input_config --shot_num 1 --wandb --cat_type cat-a --run_name 1_a")