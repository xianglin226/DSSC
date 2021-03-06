#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

module purge
conda activate /home/x/xl456/miniconda3_1/envs/torch_py38

f=./realdata/sample_151507_anno.h5

python -u run_stGAE.py --n_clusters -1 --data_file $f --save_dir sample_151507 --sample 151507 --knn 20 \
--final_labels pred.csv --final_latent_file latent.csv --run $i \
--ml_file realdata/sample_151507_mlFromMarks.txt --cl_file realdata/sample_151507_clFromMarks.txt
