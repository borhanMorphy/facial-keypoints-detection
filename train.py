import torch
import torch.nn as nn
import torch.nn.functional as F
from models import get_model
import argparse
from dataset import get_training_datasets
import torchvision.transforms as vis_transforms
from cv2 import cv2
import numpy as np

def custom_collate_fn(batch):
    imgs,targets = zip(*batch)
    imgs = torch.stack(imgs,dim=0)
    targets = torch.stack(targets,dim=0)
    return imgs,targets

def debug_collete_fn(batch):
    imgs,targets = zip(*batch)
    return imgs,targets

def show_data(imgs,targets):
    for img,target in zip(imgs,targets):
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        for x,y in target.reshape(-1,2).tolist():
            img = cv2.circle(img,(int(x),int(y)),4,(0,0,255))
        cv2.imshow("",img)
        if cv2.waitKey(0) == 27:
            exit(0)

class TargetTransform():
    def __init__(self, img_size:int=96):
        self.img_size = img_size

    def __call__(self, targets):
        return torch.from_numpy(targets).float() / self.img_size

def main(**kwargs):
    training_path = kwargs.get('training_path','./training_fixed')
    model_name = kwargs.get('model','shufflenet')
    pretrained = kwargs.get('pretrained',True)
    num_of_classes = kwargs.get('num_of_classes',30)
    device = kwargs.get('device','cuda')

    batch_size = kwargs.get('batch_size',16)
    epochs = kwargs.get('epochs',10)
    learning_rate = kwargs.get('learning_rate',5e-4)
    weight_decay = kwargs.get('weight_decay',5e-4)
    momentum = kwargs.get('momentum',0.9)

    debug = kwargs.get('debug',False)

    original_img_size = 96

    # calculated mean and std via find_sample_mean_std.py
    #mean = 0.475
    #std = 0.12

    model = get_model(model_name, pretrained=pretrained, num_of_classes=num_of_classes)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
        lr=learning_rate, weight_decay=weight_decay)


    train_transforms = val_transforms = None
    train_target_transform = val_target_transform = TargetTransform(original_img_size)
    train_transform = vis_transforms.Compose([
        vis_transforms.ToPILImage(),
        vis_transforms.Resize(model.input_size),
        vis_transforms.ToTensor()])
        #vis_transforms.Normalize(mean,std)])

    val_transform = vis_transforms.Compose([
        vis_transforms.ToPILImage(),
        vis_transforms.Resize(model.input_size),
        vis_transforms.ToTensor()])
        #vis_transforms.Normalize(mean,std)])

    if debug:
        train_target_transform = val_target_transform = None
        train_transform = val_transform = None

    train_dataset,*val_datasets = get_training_datasets(root_path=training_path,
        train_transforms=(train_transforms,train_transform,train_target_transform),
        val_transforms=(val_transforms,val_transform,val_target_transform),
        split_ratios=[0.7, 0.15, 0.15])

    val_dls = []
    train_dl = torch.utils.data.DataLoader(train_dataset,
        num_workers=4, batch_size=batch_size,
        pin_memory=True, collate_fn=custom_collate_fn if not debug else debug_collete_fn)

    for val_ds in val_datasets:
        val_dls.append( torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, num_workers=2) )

    for epoch in range(epochs):
        training_loop(train_dl,model,device,epoch,criterion,optimizer,debug=debug)
        for i,val_dl in enumerate(val_dls):
            val_loss = validation_loop(val_dl,model,device,criterion)
            print(f"[{i}+1] validation loss is :{val_loss}")

def training_loop(dl,model,device,
        epoch,criterion,optimizer,
        scheduler=None,verbose:int=10,debug:bool=False):
    model.train()
    running_loss = []
    verbose = verbose
    total_iter_size = len(dl.dataset)
    current_iter_counter = 0
    print("total data: ", len(dl.dataset))
    for imgs,targets in dl:
        if debug:
            show_data(imgs,targets)
            continue
        current_iter_counter += dl.batch_size
        optimizer.zero_grad()
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = torch.sigmoid(model(imgs))

        loss = criterion(preds,targets)

        loss.backward()
        optimizer.step()

        if scheduler: scheduler.step()

        running_loss.append(loss.item())

        if verbose == len(running_loss):
            print(f"epoch [{epoch}] iter [{current_iter_counter}/{total_iter_size}] loss: {sum(running_loss)/verbose}")
            running_loss = []

def validation_loop(dl,model,device,criterion):
    model.eval()
    running_loss = []
    for imgs,targets in dl:
        imgs = imgs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            preds = torch.sigmoid(model(imgs))
            loss = criterion(preds,targets)

        running_loss.append(loss.item())

    return sum(running_loss)/len(running_loss)

if __name__ == "__main__":
    main()