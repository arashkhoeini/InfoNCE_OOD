import torch
import argparse
import sys
import utils.utils as utils
from configs.init_configs import init_config
from pathlib import Path
from pretrainer import Trainer as Pretrainer
from datetime import datetime
import os
import numpy as np



def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main_worker(configs):
    if configs.verbose:
        print('loading imagenet data loaders')
    loaders = utils.get_loaders(configs)
    root_dir = Path(configs.dataset.root)
    if configs.verbose:
        print('loading ood data loaders')
    cub_loader = utils.get_ood_loader(configs, root_dir/'cub200_toy'/'images')
    loaders['test'] = {'cub':cub_loader,}# 'cars': cars_loader, 'products': products_loader}
    if configs.verbose: 
        print('Data loaders loaded successfully')
        print(f"training size: {len(loaders['train'].dataset)}")
        print(f"val size: {len(loaders['val'].dataset)}")
        print(f"ood cub200 size: {len(cub_loader)}")#, len(car_loader), len(products_loader))
    model = utils.get_model(configs)
    trainer = Pretrainer(model, loaders, configs)
    trainer.train()
    

def main():
    parser = argparse.ArgumentParser(description='Upper level params')
    # parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank of the current process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--dataset.root', default='', type=str, help='path to data root')
    args = parser.parse_args()
    
    if args.resume:
        if Path(args.resume).is_file():
            print("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            configs = checkpoint['configs']
            configs.resume = args.resume
        else:
            raise FileExistsError("no checkpoint found at '{}'".format(args.resume))
    else:
        configs = init_config('configs/configs.yml', args)

    configs.output_path = os.path.join('output', datetime.now().strftime("%m-%d-%H-%M")) 
    utils.mkdir(configs.output_path)
    if configs.seed: init_seed(configs.seed)
    main_worker(configs)
        

if __name__ == '__main__':
    """
    This snippet is the main file for pretraining. It is responsible for loading the model, the dataset, and the optimizer.
    If a checkpoint is provided as --resume parameter, it will load the checkpoint and resume training from that point.
    In order to run the script on muliple GPUs, you need to first set CUDA: true and then run the script using the following command:
    `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE --use_env main_pretrain.py`
    """
    main()
    