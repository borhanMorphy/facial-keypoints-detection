import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import get_backbone, get_available_backbones
from losses import get_criterion, get_available_criterions
from optimizers import get_optimizer, get_available_optimizers
from model import FacialKeypointsDetector

from dataset import get_training_datasets

import os
import argparse
import json
from typing import Dict

from cv2 import cv2
import numpy as np
import transforms
import math
from utils import save_checkpoint,load_checkpoint,seed_everything
# TODO write docstring

def parse_json(file_path:str) -> Dict:
    with open(file_path,"r") as foo:
        data = json.load(foo)
    return data

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs','-c',type=parse_json,default='./configs/default.json')
    ap.add_argument('--training-data-path',type=str)
    ap.add_argument('--checkpoint-path',type=str)

    ap.add_argument('--backbone',type=str,choices=get_available_backbones())
    ap.add_argument('--criterion',type=str,choices=get_available_criterions())
    ap.add_argument('--optimizer',type=str,choices=get_available_optimizers())
    # TODO 
    # ap.add_argument('--scheduler',type=str,choices=)

    ap.add_argument('--pretrained',action='store_true')
    ap.add_argument('--num-classes',type=int)
    ap.add_argument('--device',type=str,choices=['cpu','cuda'])

    ap.add_argument('--batch-size',type=int)
    ap.add_argument('--epochs',type=int)
    ap.add_argument('--verbose',type=int)

    ap.add_argument('--train-split',type=float)
    ap.add_argument('--nfolds',type=int)

    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--seed',type=int)

    kwargs = vars(ap.parse_args())
    configs = kwargs.pop('configs')
    
    for k,v in kwargs.items():
        if not isinstance(v,type(None)): configs[k] = v
    return configs

def custom_collate_fn(batch):
    imgs,targets = zip(*batch)
    imgs = torch.stack(imgs,dim=0)
    targets = torch.stack(targets,dim=0)
    return imgs,targets

class TargetTransform():
    def __init__(self, img_size:int=96):
        self.img_size = img_size

    def __call__(self, targets):
        return torch.from_numpy(targets).float() / self.img_size

def main(**kwargs):
    # TODO add tensorboard
    training_path = kwargs.get('training_data_path')
    checkpoint_path = kwargs.get('checkpoint_path')
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    backbone_name = kwargs.get('backbone')
    criterion_name = kwargs.get('criterion').upper()
    optimizer_name = kwargs.get('optimizer').upper()
    scheduler = kwargs.get('scheduler',None)

    pretrained = kwargs.get('pretrained')
    num_classes = kwargs.get('num_classes')
    device = kwargs.get('device')

    batch_size = kwargs.get('batch_size')
    epochs = kwargs.get('epochs')
    hyperparameters = kwargs.get('hyperparameters',{})
    verbose = kwargs.get('verbose')

    train_split = kwargs.get('train_split')
    nfolds = kwargs.get('nfolds')

    val_splits = [(1-train_split) / nfolds] * nfolds

    resume = kwargs.get('resume')

    seed = hyperparameters.get('seed')

    if seed: seed_everything(seed)
    
    # TODO calculate mean and std
    mean = hyperparameters.get('mean',0)
    std = hyperparameters.get('std',1)

    splits = [train_split]+val_splits
    assert sum(splits) <= 1,"given splits must be lower or equal than 1"

    original_img_size = 96

    criterion = get_criterion(criterion_name)

    backbone = get_backbone(backbone_name, pretrained=pretrained)

    model = FacialKeypointsDetector(backbone, criterion=criterion,
        device=device, num_classes=num_classes)

    optimizer = get_optimizer(optimizer_name, model.parameters(),
        kwargs=hyperparameters.get('optimizer',{}))

    train_transforms = val_transforms = None
    train_target_transform = val_target_transform = TargetTransform(original_img_size)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(model.get_input_size()),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(model.get_input_size()),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    train_dataset,*val_datasets = get_training_datasets(root_path=training_path,
        train_transforms=(train_transforms,train_transform,train_target_transform),
        val_transforms=(val_transforms,val_transform,val_target_transform),
        split_ratios=splits)

    val_dls = []
    train_dl = torch.utils.data.DataLoader(train_dataset,
        num_workers=4, batch_size=batch_size,
        pin_memory=True, collate_fn=custom_collate_fn)

    for val_ds in val_datasets:
        val_dls.append( torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, num_workers=2) )

    current_epoch = 0
    best_loss = math.inf
    if resume:
        print(f"loading checkpoint from {checkpoint_path}")
        best_loss,current_epoch = load_checkpoint(model, optimizer, scheduler=scheduler,
            save_path=checkpoint_path, suffix='last')

    try:
        for epoch in range(current_epoch,epochs):
            training_loop(train_dl, model, epoch, epochs, optimizer,
                scheduler=scheduler, verbose=verbose)

            val_losses = []
            for i,val_dl in enumerate(val_dls):
                val_loss = validation_loop(val_dl, model)
                val_losses.append(val_loss)
                print(f"validation [{i+1}] loss:  {val_loss:.07f}")

            mean_val_loss = sum(val_losses) / len(val_losses)
            print(f"validation [mean] loss:  {mean_val_loss:.07f}")
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                print("saving best checkpoint...")
                save_checkpoint(model,optimizer,epoch,best_loss,
                    scheduler=scheduler, suffix='best', save_path=checkpoint_path)

            print("saving last checkpoint...")
            save_checkpoint(model,optimizer,epoch,best_loss,
                scheduler=scheduler, suffix='last', save_path=checkpoint_path)

    except KeyboardInterrupt:
        print("training interrupted with ctrl+c saving current state of the model")
        save_checkpoint(model,optimizer,epoch,best_loss,
            scheduler=scheduler, suffix='last', save_path=checkpoint_path)

def training_loop(dl,model,epoch,epochs,optimizer,
        scheduler=None,verbose:int=10,debug:bool=False):
    running_loss = []
    verbose = verbose
    total_iter_size = len(dl.dataset)
    current_iter_counter = 0
    for imgs,targets in dl:
        current_iter_counter += dl.batch_size
        optimizer.zero_grad()

        loss = model.training_step(imgs,targets)

        loss.backward()
        optimizer.step()

        if scheduler: scheduler.step()

        running_loss.append(loss.item())

        if verbose == len(running_loss):
            print(f"epoch [{epoch+1}/{epochs}]  iter [{current_iter_counter}/{total_iter_size}]  loss: {sum(running_loss)/verbose:.07f}")
            running_loss = []

def validation_loop(dl,model):
    running_loss = []
    for imgs,targets in dl:
        loss = model.val_step(imgs,targets)
        running_loss.append(loss)

    return sum(running_loss)/len(running_loss)

if __name__ == "__main__":
    kwargs = parse_arguments()
    print(json.dumps(kwargs,sort_keys=False,indent=4))
    main(**kwargs)