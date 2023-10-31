#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-00:05            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti-dev # Partition to submit to
#SBATCH --gres=gpu:4              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/job_%j.out  # File to which STDOUT will be written
#SBATCH --error=logs/job_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<your-email>  # Email to which notifications will be sent

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
echo -e "---------------------------------\n"

# Due to a potential bug, we need to manually load our bash configurations first
source /mnt/qb/home/baumgartner/sun22/.bashrc

cd /mnt/qb/work/baumgartner/sun22/project/tmi

# Next activate the conda environment
conda activate tt_interaction


echo "convert training images to png"
python3 ./data/preprocess_vindr.py \
  --input-dir "/mnt/qb/baumgartner/rawdata/vindr-cxr-physionet-original/1.0.0/train" \
  --output-dir "/mnt/qb/baumgartner/rawdata/vindr-cxr-physionet-pngs/1.0.0/train" \
  --cpus 4 \
  --log-file "/mnt/qb/baumgartner/rawdata/vindr-cxr-physionet-pngs/1.0.0/convert_train_log.txt" \
  --out-file "/mnt/qb/baumgartner/rawdata/vindr-cxr-physionet-pngs/1.0.0/convert_train_results.csv" \

python3 spine/preprocess/dicom2png.py \
  --input-dir "/mnt/qb/baumgartner/rawdata/vindr-cxr-physionet-original/1.0.0/test" \
  --output-dir "/mnt/qb/baumgartner/rawdata/vindr-cxr-physionet-pngs/1.0.0/test" \
  --cpus 4 \
  --log-file "/mnt/qb/baumgartner/rawdata/vindr-cxr-physionet-pngs/1.0.0/convert_test_log.txt" \
  --out-file "/mnt/qb/baumgartner/rawdata/vindr-cxr-physionet-pngs/1.0.0/convert_test_results.csv" \

# Deactivate environment again
conda deactivate


