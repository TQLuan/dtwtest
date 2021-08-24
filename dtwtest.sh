#!/bin/bash
#
#SBATCH --job-name=moverscore
#SBATCH --output=/ukp-storage-1/ychen/dtwtest.txt
#SBATCH --mail-user=chenyr1996@hotmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/ychen/py36/bin/activate
module load cuda/10.0
python -u /ukp-storage-1/ychen/dtwtest/main.py
