#!/bin/bash --login
#$ -cwd
#$ -l a100=1
#$ -t 1
#$ -l s_rt=24:00:00

echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"

module load apps/binapps/anaconda3/2021.11
module swap tools/env/proxy tools/env/proxy2
module load compilers/gcc/9.3.0 # Needed for geometric_kernels package
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113 # Loads correct CUDA, works for V100s and A100s
# conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
# 'cd scratch' followed by 'conda create --prefix=point_clouds python=3.9', activate with....
conda activate /net/scratch2/c75022tm/gps-for-point-clouds/point_clouds

# Args: 2k, 10k, Armadillo, 50, 50, 500 to match Stuti's CPU test
