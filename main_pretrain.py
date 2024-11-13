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
    loaders['test'] = {}
    for ood_dataset in configs.dataset.ood:
        loader = utils.get_ood_loader(configs, root_dir/ood_dataset)
        loaders['test'][ood_dataset] = loader
        if configs.verbose: 
            print('Data loaders loaded successfully')
            print(f"training size: {len(loaders['train'].dataset)}")
            print(f"val size: {len(loaders['val'].dataset)}")
            print(f"{ood_dataset} size: {len(loader)}")#, len(car_loader), len(products_loader))
    model = utils.get_model(configs)
    trainer = Pretrainer(model, loaders, configs)
    trainer.train()
    

def main():
    parser = argparse.ArgumentParser(description='PyTorch Project Configurations')

    # General Configurations
    parser.add_argument('--verbose', default=True, type=bool, help='enable verbose logging')
    parser.add_argument('--cuda', default=False, type=bool, help='use CUDA if available')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='rank of the current process')
    parser.add_argument('--seed', default=1, type=int, help='random seed for reproducibility')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint to resume training')

    # WandB Configurations
    parser.add_argument('--wandb.enable', default=False, type=bool, help='enable Weights & Biases logging')
    parser.add_argument('--wandb.project', default='resnet50-infonce-imagenet', type=str, help='Weights & Biases project name')
    parser.add_argument('--wandb.key', default='7047c70c12cc7631dfdbc7f66f14b27e9d06c71d', type=str, help='Weights & Biases API key')

    # Dataset Configuration
    parser.add_argument('--dataset.name', default='miniimagenet_toy', type=str, help='name of the dataset')
    parser.add_argument('--dataset.root', default='$SLURM_TMPDIR/data', type=str, help='path to dataset root directory')
    parser.add_argument('--dataset.ood', default=['cub200_toy/images'], type=list, help='list of out-of-distribution datasets')
    parser.add_argument('--dataset.batch_size', default=64, type=int, help='batch size for training and validation')

    # Model Configuration
    parser.add_argument('--model.method', default='supervised_infonce', choices=['supervised_infonce', 'infonce', 'mocov2'],type=str, help='training method (e.g., mocov2 or infonce)')
    parser.add_argument('--model.encoder', default='resnet50', type=str, help='encoder architecture')
    parser.add_argument('--model.checkpoint_source', default='microsoft', choices=['microsoft', 'leftthomas', 'francesco'], type=str, help='source for pre-trained checkpoints')
    parser.add_argument('--model.temperature', default=0.07, type=float, help='temperature parameter for contrastive learning')
    parser.add_argument('--model.feature_dim', default=128, type=int, help='dimensionality of the feature embedding')
    parser.add_argument('--model.num_classes', default=100, type=int, help='number of classes in the dataset')
    parser.add_argument('--model.input_size', default=[3, 32, 32], type=list, help='input image dimensions')
    parser.add_argument('--model.pretrained', default=True, type=bool, help='use pretrained model weights')

    # Pretraining Configuration
    parser.add_argument('--pretraining.epochs', default=1, type=int, help='number of pretraining epochs')
    parser.add_argument('--pretraining.warmup_epochs', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--pretraining.optimizer', default='lamb', type=str, choices=['lamb', 'sgd'],  help='optimizer type (e.g., lamb or std)')
    parser.add_argument('--pretraining.learning_rate', default=0.001, type=float, help='learning rate for optimizer')
    parser.add_argument('--pretraining.weight_decay', default=0.0001, type=float, help='weight decay for optimizer')
    parser.add_argument('--pretraining.momentum', default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument('--pretraining.checkpoint_freq', default=1, type=int, help='frequency of checkpoint saving')
    parser.add_argument('--pretraining.accumulation_steps', default=1, type=int, help='number of gradient accumulation steps')

    # Training Configuration
    parser.add_argument('--training.epochs', default=1, type=int, help='number of training epochs')
    parser.add_argument('--training.warmup_epochs', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--training.learning_rate', default=0.001, type=float, help='learning rate for training')
    parser.add_argument('--training.weight_decay', default=0.0001, type=float, help='weight decay for training optimizer')
    parser.add_argument('--training.momentum', default=0.9, type=float, help='momentum for training optimizer')
    parser.add_argument('--training.checkpoint_freq', default=1, type=int, help='frequency of checkpoint saving during training')

    # Logging Configuration
    parser.add_argument('--logging.log_dir', default='/path/to/logs', type=str, help='directory to save log files')
    parser.add_argument('--logging.tensorboard', default=True, type=bool, help='enable TensorBoard logging')
    parser.add_argument('--logging.checkpoint', default=True, type=bool, help='enable checkpoint saving')
    
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
    