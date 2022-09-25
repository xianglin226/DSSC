#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -J dssctest

module purge
conda activate /home/x/xl456/miniconda3_1/envs/torch_py38

f=../sample_data/sample_151507_anno.h5

for i in {1..5}
 do
python -u run_DSSC.py --n_clusters -1 --data_file $f \
--save_dir out_151507 --knn 20 \
--final_labels pred.csv --final_latent_file latent.csv --run $i \
--ml_file sample_151507_mlFromMarks.txt --cl_file sample_151507_clFromMarks.txt
 done
