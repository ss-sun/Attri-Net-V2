#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-72:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=20G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/baumgartner/sun22/logs/hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/baumgartner/sun22/logs/hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<susu.sun@uni-tuebingen.de>  # Email to which notifications will be sent
#SBATCH --array=25,30,34,40   #

# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here


# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID 
echo -e "---------------------------------\n"


# Due to a potential bug, we need to manually load our bash configurations first
source /mnt/qb/home/baumgartner/sun22/.bashrc

# cd /mnt/qb/work/baumgartner/sun22/project/tmi
cd /mnt/qb/work/baumgartner/sun22/github_projects/tmi

# Next activate the conda environment 
conda activate tt_interaction


# Run our code
echo "-------- PYTHON OUTPUT ----------"
# python3 main.py --use_wandb False
# python3 main.py --lambda_1 100 --lambda_2 300 --lambda_3 100 --use_wandb False
# python3 main.py --lambda_1 500 --lambda_2 1000 --lambda_3 500 --use_wandb True
# python3 main.py --lambda_1 200 --lambda_2 400 --lambda_3 200
# python3 main.py --dataset contam50
# python3 main.py
# python3 main.py --manual_seed 12345
# python3 main.py --manual_seed 2023
# python3 main.py --manual_seed 8675309
# python3 main.py --manual_seed 21
# python3 main.py --manual_seed 4294438

# python3 main_attrinet.py --debug "False" --dataset_idx ${SLURM_ARRAY_TASK_ID} --use_wandb "False" # train attrinet on gray and color glocuma images
# python3 main_attrinet.py --debug "False" --dataset_idx ${SLURM_ARRAY_TASK_ID} --lambda_3 0 --lambda_centerloss 0 --lambda_localizationloss 0 --use_wandb "False" # train attrinet on gray and color glocuma images wihtout classifiers loss

# python3 main_attrinet.py --debug "False" --dataset_idx ${SLURM_ARRAY_TASK_ID} --use_wandb "False"

python3 main_attrinet.py --debug "False" --dataset_idx 0 --lambda_localizationloss ${SLURM_ARRAY_TASK_ID} --use_wandb "False"


echo "---------------------------------"


# Deactivate environment again
conda deactivate

