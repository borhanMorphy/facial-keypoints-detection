import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import get_backbone
from losses import get_criterion
from optimizers import get_optimizer
from model import FacialKeypointsDetector

import argparse
from dataset import get_training_datasets
from cv2 import cv2
import numpy as np
import transforms
import math

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
    training_path = kwargs.get('training_data_path','./data/training_fixed')

    backbone_name = kwargs.get('backbone','shufflenet')
    criterion_name = kwargs.get('criterion','MSE').upper()
    optimizer_name = kwargs.get('optimizer','ADAM').upper()

    pretrained = kwargs.get('pretrained',True)
    num_classes = kwargs.get('num_classes',30)
    device = kwargs.get('device','cuda')

    batch_size = kwargs.get('batch_size',16)
    epochs = kwargs.get('epochs',10)
    hyperparameters = kwargs.get('hyperparameters', {})

    train_split = kwargs.get('train_split',0.7)
    val_splits = kwargs.get('val_splits',[0.15, 0.15])

    # TODO add resume
    resume = kwargs.get('resume',True)
    # TODO add seed
    seed = kwargs.get('seed',-1)

    # TODO calculate mean and std
    mean = kwargs.get('mean',0)
    std = kwargs.get('std',1)
    best_loss = math.inf

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

    for epoch in range(epochs):
        training_loop(train_dl, model, epoch, optimizer)

        val_losses = []
        for i,val_dl in enumerate(val_dls):
            val_loss = validation_loop(val_dl, model)
            val_losses.append(val_loss)
            print(f"[{i}+1] validation loss is: {val_loss:.04f}")

        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f"[mean] validation loss is: {mean_val_loss:.04f}")
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            # TODO best loss achived, save state of the model as best.pt
            pass

        # TODO after each epoch save loss as last.pt

def training_loop(dl,model,epoch,optimizer,
        scheduler=None,verbose:int=10,debug:bool=False):
    # TODO add scheduler

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
            print(f"epoch [{epoch}] iter [{current_iter_counter}/{total_iter_size}] loss: {sum(running_loss)/verbose}")
            running_loss = []

def validation_loop(dl,model):
    running_loss = []
    for imgs,targets in dl:
        loss = model.val_step(imgs,targets)
        running_loss.append(loss)

    return sum(running_loss)/len(running_loss)

if __name__ == "__main__":
    main()