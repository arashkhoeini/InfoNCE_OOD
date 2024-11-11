#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-ester
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akhoeini@sfu.ca

cd '/home/akhoeini/scratch/data'
zip -r miniimagenet.zip miniimagenet/