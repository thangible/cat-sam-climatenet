import os
import time
os.system("python train.py --config input_config --shot_num 16 --wandb --cat_type cat-a")
time.sleep(20)
os.system("python train.py --config input_config --shot_num 16 --wandb --cat_type cat-t")
time.sleep(20)
os.system("python train.py --config input_config --shot_num 1 --wandb --cat_type cat-t")
time.sleep(20)
os.system("python train.py --config input_config  --wandb --cat_type cat-t")
time.sleep(20)
os.system("python train.py --config input_config  --wandb --cat_type cat-a")