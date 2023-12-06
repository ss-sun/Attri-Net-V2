from __future__ import print_function
import os
import shutil
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import wandb
import datetime



def init_seed(manual_seed):
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    cudnn.benchmark = True

def create_expname(exp_configs):
    current_time = datetime.datetime.now()

    if exp_configs.exp_name == "resnet" or exp_configs.exp_name == "bcos_resnet":
        exp_name = exp_configs.exp_name + str(current_time)[:-7]
        exp_name += f"-{exp_configs.dataset}"
        exp_name += f"-bs={exp_configs.batch_size}"
        exp_name += f"-lr={exp_configs.lr}"
        exp_name += f"-weight_decay={exp_configs.weight_decay}"
        if exp_configs.exp_name == "bcos_resnet":
            exp_name += f"-lambda_localizationloss={exp_configs.lambda_localizationloss}"



    if exp_configs.exp_name == "attri-net":
        exp_name = exp_configs.exp_name + str(current_time)[:-7]
        exp_name += f"--{exp_configs.dataset}"
        exp_name += f"--process_mask={exp_configs.process_mask}"
        # exp_name += f"--bs={exp_configs.batch_size}"
        exp_name += f"--lg_ds={exp_configs.lgs_downsample_ratio}"
        exp_name += f"--l_cri={exp_configs.lambda_critic}"
        exp_name += f"--l1={exp_configs.lambda_1}"
        exp_name += f"--l2={exp_configs.lambda_2}"
        exp_name += f"--l3={exp_configs.lambda_3}"
        exp_name += f"--l_ctr={exp_configs.lambda_centerloss}"
        if exp_configs.lambda_localizationloss > 0:
            exp_name += f"--l_loc={exp_configs.lambda_localizationloss}"
            exp_name += f"--guid_freq={exp_configs.guidance_freq}"
        exp_name += f"--seed={exp_configs.manual_seed}"

    return exp_name




def init_experiment(exp_configs):
    exp_configs.exp_name = create_expname(exp_configs)
    print('exp_configs.exp_name', exp_configs.exp_name)
    os.makedirs(exp_configs.save_path, exist_ok=True)
    exp_configs.exp_dir = os.path.join(exp_configs.save_path, exp_configs.exp_name)
    exp_configs.ckpt_dir = exp_configs.exp_dir + '/ckpt'
    exp_configs.output_dir = exp_configs.exp_dir + '/output'

    for path in [exp_configs.exp_dir, exp_configs.ckpt_dir, exp_configs.output_dir]:
        try:
            shutil.rmtree(path)
        except:
            pass
        os.makedirs(path)


def wandb_set_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ['WANDB__SERVICE_WAIT'] = f'{seconds}'



def init_wandb(exp_configs):
    # Set up wandb.
    wandb.login(key='34f6af5c2b35419f4b738daa31203a144ca49987') # Set your wandb key
    wandb_set_startup_timeout(1200)
    wandb.init(dir=exp_configs.save_path,
               project="Attri-Net TMI exps",
               name = exp_configs.exp_name,
               notes='train on' + exp_configs.dataset,
               )

    config = wandb.config
    if "resnet" in exp_configs.exp_name:
        config.batch_size = exp_configs.batch_size
        config.lr = exp_configs.lr
    if "attri-net" in exp_configs.exp_name:
        config.logreg_mode = exp_configs.lgs_downsample_ratio
        config.lambda_critic = exp_configs.lambda_critic
        config.lambda_1 = exp_configs.lambda_1
        config.lambda_2 = exp_configs.lambda_2
        config.lambda_3 = exp_configs.lambda_3
        config.lambda_centerloss = exp_configs.lambda_centerloss

    wandb.run.save()


