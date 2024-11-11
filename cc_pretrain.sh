#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=def-ester
# #SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akhoeini@sfu.ca

module load StdEnv/2020  gcc/9.3.0  cuda/11.4
module load faiss/1.7.3

export MASTER_ADDR=localhost
export MASTER_PORT=12355

DATA_DIR='/project/6007580/data/imagenet/'

original_dir=$(pwd)

cd $SLURM_TMPDIR
mkdir data
cd data 

mkdir train && cp ${DATA_DIR}ILSVRC2012_img_train.tar train/ && cd train
tar -xf ILSVRC2012_img_train.tar -C $SLURM_TMPDIR/data/train --checkpoint=.50000
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xf "${NAME}" -C "${NAME%.tar}" --checkpoint=.5000; rm -f "${NAME}"; done
rm -r ILSVRC2012_img_train/
cd ..
# Extract the validation data and move images to subfolders:
mkdir val && cp ${DATA_DIR}hf_val_images.tar.gz val/ && cd val && tar -xzf hf_val_images.tar.gz
# rm -r hf_val_images/ 
# cd ..
# ls train/
# ls val/
cd $original_dir

bash cc_downstream_extract.sh

# /localscratch/akhoeini.44716129.0/data/val/main_pretrain.py
ls $SLURM_TMPDIR
ls $SLURM_TMPDIR/ood-data
ls $SLURM_TMPDIR/data
ls $SLURM_TMPDIR/data/train
ls $SLURM_TMPDIR/data/val
python main_pretrain.py --dataset.root $SLURM_TMPDIR/data