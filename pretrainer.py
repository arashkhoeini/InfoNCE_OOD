import torch.distributed as dist
import torch
import argparse
import sys
import time
import utils.utils as utils
from configs.init_configs import init_config
import torch.optim as optim
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.lamb import Lamb
from loss.infonce import SupervisedInfoNCE, InfoNCE
import wandb
from utils.metrics import NearestNeighboursMetrics
from torch.optim.lr_scheduler import CosineAnnealingLR

class Trainer():

    def __init__(self, model, loaders, configs):
        self.model = model
        self.configs = configs
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
        self.wandb_enable = self.configs.wandb.enable and (self.configs.rank == 0)
        self.device = f'cuda:{self.configs.rank}' if self.configs.cuda else f'cpu'
        self.model = self.model.to(self.device)
        if self.configs.cuda:
            if self.configs.world_size > 1:
                if self.configs.verbose: print(f"Setting the model to train on {self.configs.world_size} GPUs")
                # Initialize the process group
                dist.init_process_group(backend='nccl', init_method='env://', world_size=self.configs.world_size, rank=self.configs.rank)
                self.model.to(self.device)
                self.model = DDP(self.model, device_ids=[self.configs.rank])
            else:
                self.model.to(self.device)
        else:
            self.model.to(self.device)
        
        if self.wandb_enable:
            wandb.login(key=self.configs.wandb.key)
            wandb.init(project=self.configs.wandb.project)

    def train(self):
        global_batch_size = self.configs.dataset.batch_size * self.configs.world_size * self.configs.pretraining.accumulation_steps
        lr_scaled = self.configs.pretraining.learning_rate * math.sqrt(global_batch_size / 256)
        if self.configs.pretraining.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                lr=lr_scaled, 
                                momentum=self.configs.pretraining.momentum, 
                                weight_decay=self.configs.pretraining.weight_decay)
        elif self.configs.pretraining.optimizer == 'lamb':
            optimizer = Lamb(self.model.parameters(),
                                lr=lr_scaled, 
                                weight_decay=self.configs.pretraining.weight_decay)
        else:
            raise NotImplementedError(f'optimizer {self.configs.pretraining.optimizer} is unknown')
        cosine_scheduler = CosineAnnealingLR(optimizer, 
                                             T_max=(self.configs.pretraining.epochs - self.configs.pretraining.warmup_epochs))

        if self.configs.model.method == 'supervised_infonce':
            criterion = SupervisedInfoNCE(temperature=self.configs.model.temperature)
        elif self.configs.model.method == 'infonce':
            criterion = InfoNCE(temperature=self.configs.model.temperature)
        elif self.configs.model.method == 'mocov2':
            criterion = torch.nn.CrossEntropyLoss()  

        start_epoch = 0
        # use Pathlib to check if resume is a file
        
        if self.configs.resume:
            if self.configs.verbose: print("loading checkpoint '{}'".format(self.configs.resume))
            checkpoint = torch.load(self.configs.resume).to(self.device)
            start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    self.configs.resume, checkpoint["epoch"]
                )
            )
        # start pretrining
        for epoch in range(start_epoch, self.configs.pretraining.epochs):
            if self.configs.world_size > 1: self.train_loader.sampler.set_epoch(epoch)
            if epoch < self.configs.pretraining.warmup_epochs:
                self._adjust_learning_rate(optimizer, epoch, 0.1, self.configs.pretraining.warmup_epochs)
            else:
                cosine_scheduler.step()
            train_results = self._train_epoch(self.train_loader, criterion, optimizer, epoch)
            state = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "configs": self.configs,
                    "optimizer": optimizer.state_dict(),
                }
            if (self.configs.rank == 0) and (epoch % self.configs.pretraining.checkpoint_freq) == 0:
                test_results = self.evaluate()
                if self.wandb_enable:
                    wandb.log({**train_results, **test_results})
                self.save_checkpoint(state, False)

    def _train_epoch(self, train_loader, criterion, optimizer, epoch):
        batch_time = utils.AverageMeter("Time", ":6.3f")
        data_time = utils.AverageMeter("Data", ":6.3f")
        losses = utils.AverageMeter("Loss", ":.4e")
        
        self.model.train()
        end = time.time()
        if self.configs.verbose: print("pretraining starts")
        for i, (original_image, image_aug1, image_aug2, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            image_aug1 = image_aug1.to(self.device)
            image_aug2 = image_aug2.to(self.device)
            target = target.to(self.device)
            
            if self.configs.model.method == 'supervised_infonce':
                representations = self.model(image_aug1)
                loss = criterion(representations, target)
                # recall1, recall5 = self.nn_recall(representations)

            elif self.configs.model.method == 'mocov2':
                output, target = self.model(im_q=image_aug1, im_k=image_aug2)
                filtered_output = self._filter_logits_based_on_difficulty(output)
                loss = criterion(filtered_output, target)
            
            # measure accuracy and record loss
            
            losses.update(loss.item(), image_aug1.size(0))

            # compute gradient and do SGD step
            # warmup epochs are only used for MoCo in order to fill the queue

            loss.backward()
            if self.configs.model.method == 'mocov2':
                if epoch +1 > self.configs.pretraining.warmup_epochs:
                    if (epoch+1)%self.configs.pretraining.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad() 
            else:
                if (epoch+1)%self.configs.pretraining.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #TODO: DELETE THIS
            print(f"Epoch {epoch}/{self.configs.pretraining.epochs} - Batch {i}/{len(train_loader)} | loss {losses.avg} - Batch Time: {batch_time.avg:.3f} seconds, Data Time: {data_time.avg:.3f} seconds ")
            break
        # EPOCH DONE!
            
        # Gather timing information from all processes
        if self.configs.world_size > 1:
            if self.configs.rank == 0:
                batch_time_tensor = torch.tensor([batch_time.avg], device=self.device)
                data_time_tensor = torch.tensor([data_time.avg], device=self.device)
                loss_tensor = torch.tensor([losses.avg], device=self.device)

                dist.all_reduce(batch_time_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(data_time_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    
                total_batch_time = batch_time_tensor.item() / dist.get_world_size()
                total_data_time = data_time_tensor.item() / dist.get_world_size()
                mean_loss = data_time_tensor.item() / dist.get_world_size()
        else:
            total_batch_time = batch_time.avg
            total_data_time = data_time.avg
            mean_loss = losses.avg

        if self.configs.rank == 0:  # Only print from the main process
            print(f"loss {mean_loss} - "
                f"Batch Time: {total_batch_time:.3f} seconds, "
                f"Data Time: {total_data_time:.3f} seconds, ")
        
        results = {"mean_loss": mean_loss, "batch_time": total_batch_time, "data_time": total_data_time}#, "train_recall1": recall1, "train_recall5": recall5}
        return results
    
    def __del__(self):
        if self.wandb_enable:
            wandb.finish()
        if self.configs.cuda and (self.configs.world_size > 1):
            self._cleanup()

    def _compute_feature_space_metrics(self):
        self.model.eval()
        with torch.no_grad():
            for i, (original_image, image_aug1, image_aug2, _) in enumerate(self.val_loader):
                image_aug1 = image_aug1.to(self.device)
                image_aug2 = image_aug2.to(self.device)
                original_image = original_image.to(self.device)
                
                aug1_reps = self.model.get_representations(image_aug1)
                aug2_reps = self.model.get_representations(image_aug2)
                original_reps = self.model.get_representations(original_image)

                if i == 0:
                    all_aug1 = aug1_reps
                    all_aug2 = aug2_reps
                    all_original = original_reps
                else:
                    all_aug1 = torch.cat((all_aug1, aug1_reps), dim=0)
                    all_aug2 = torch.cat((all_aug2, aug2_reps), dim=0)
                    all_original = torch.cat((all_original, original_reps), dim=0)
        
        uniformity = utils.uniform_loss(all_original)
        alignment = utils.align_loss(all_aug1, all_aug2)

        pairwise_distances = torch.cdist(all_original, all_original, p=2)

        entropy = utils.entropy_loss(all_original, pairwise_distances)
        recall_at_1 = utils.recall_at_1(all_original, pairwise_distances)

        return {'uniformity': -uniformity, 'alignment': -alignment, 'entropy': entropy, 'recall_at_1': recall_at_1}

    def save_checkpoint(self, state, is_best):
        # TODO: impelement the is_best logic
        epoch = state['configs'].training.epochs
        filename = Path(self.configs.output_path) / f"pretraining_checkpoint_{epoch}.pth.tar"
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "pretraining_best.pth.tar")    

    def evaluate(self):
        nn_metrics = NearestNeighboursMetrics()

        results = {}
        self.model.eval()
        with torch.no_grad(): 
            if self.configs.verbose: print('Evaluation: starts')
            # all_reps = []
            # all_targets = []
            # for images, _, _, targets in self.train_loader:
            #     images = images.to(self.device)
            #     targets = targets.to(self.device)
            #     reps = self.model(images)
                
            #     all_reps.append(reps)
            #     all_targets.append(targets)
            # all_reps = torch.cat(all_reps, dim=0)
            # all_targets = torch.cat(all_targets, dim=0)
            # metrics = nn_metrics(all_reps, all_targets)
            # results.update({f'train_{key}':value for key,value in metrics.items()})

            all_reps = []
            all_targets = []
            if self.configs.verbose: print('Evaluation: computing val images representations...')
            for images, _, _, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                reps = self.model(images)
                
                all_reps.append(reps)
                all_targets.append(targets)
                break
            all_reps = torch.cat(all_reps, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            if self.configs.verbose: print('Evaluation: computing Validation NN metrics...')
            metrics = nn_metrics(all_reps, all_targets)
            results.update({f'val_{key}':value for key,value in metrics.items()})

            if self.configs.verbose: print('Evaluation: computing test images representations and NN metrics')
            for dataset in self.test_loader.keys():
                loader = self.test_loader[dataset]
                all_reps = []
                all_targets = []
                # get all the representations of all the samples in loader
                for images, targets in loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    reps = self.model(images)
                    
                    all_reps.append(reps)
                    all_targets.append(targets)
                    break
                all_reps = torch.cat(all_reps, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                metrics = nn_metrics(all_reps, all_targets)
            results.update({f'{dataset}_{key}':value for key,value in metrics.items()})
            if self.configs.verbose: print('Evaluation: ends')
        return results

    def _adjust_learning_rate(self, optimizer, epoch, base_lr, warmup_epochs):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def _cleanup():
        dist.destroy_process_group()



