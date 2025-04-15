#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu.q
#$ -l gpu=1
source /usr/local/gpuallocation.sh
conda activate gpuenv
python mainkeel.py wdbc