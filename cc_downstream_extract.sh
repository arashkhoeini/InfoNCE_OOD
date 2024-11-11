#!/bin/bash
# 
# script to extract URL downstream datasets:
# 
# cub
# └───images
# |    └───BMW 3 Series Wagon 2012
# |           │   00039.jpg
# |           │   ...
# |    ...
# cars
# └───images
# |    └───001.Black_footed_Albatross
# |           │   Black_Footed_Albatross_0001_796111.jpg
# |           │   ...
# |    ...
# online_products
# └───images
# |    └───bicycle_final
# |           │   111085122871_0.jpg
# |    ...
# |
# └───Info_Files
# |    │   bicycle.txt
# |    │   ...
#
#
# Extract the data:
#

DATA_DIR='/project/6007580/data/'

cd $SLURM_TMPDIR
mkdir ood-data
cd ood-data

# pv ${DATA_DIR}cars196.tar | tar -xf - -C $SLURM_TMPDIR/data-downstream/ 
echo -e 'untar cars196'
cp ${DATA_DIR}cars196.tar . && tar -xf cars196.tar -C .  --checkpoint=.1000
mv cars196 cars

echo -e '\n\nuntar cub200'
cp ${DATA_DIR}cub200.tar . && tar -xf cub200.tar -C .  --checkpoint=.1000
mv cub200 cub
rm cub200.tar

echo -e '\n\nuntar online_products'
cp ${DATA_DIR}online_products.tar . && tar -xf online_products.tar -C .  --checkpoint=.1000
rm online_products.tar

echo -e '\ndone extracting downstream datasets\n\n'
