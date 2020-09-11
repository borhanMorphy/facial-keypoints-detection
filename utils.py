from typing import Tuple
import numpy as np
from cv2 import cv2
import os
from collections import OrderedDict
import torch
import random

def str2img(str_img:str, img_size:Tuple=(96,96)) -> np.ndarray:
    """converts string image to numpy image"""
    return np.array([int(pixel) for pixel in str_img.split(" ")]).astype(np.uint8).reshape(img_size)

def load_img(img_path:str) -> np.ndarray:
    """loads image with given file path"""
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

def visualize(img:np.ndarray, keypoints:np.ndarray) -> int:
    """show image with keypoints

    Args:
        img (np.ndarray): h,w or h,w,3 channeled image
        keypoints (np.ndarray): N,2 keypoints

    Returns:
        int: clicked cv2.waitKey key
    """
    simg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    assert len(keypoints.shape) == 2 and keypoints.shape[-1] == 2,"wrong keypoints format"
    for x,y in keypoints.astype(np.int32):
        simg = cv2.circle(simg, (x,y), (2), (0,0,255))
    cv2.imshow("", simg)
    return cv2.waitKey(0)

def seed_everything(magic_number:int):
    assert isinstance(magic_number,int),f"magic number type is incorrect, found:{type(magic_number)} but expected integer"
    random.seed(magic_number)
    torch.manual_seed(magic_number)
    np.random.seed(magic_number)

def save_checkpoint(model, optimizer, epoch:int, best_loss:float,
        scheduler=None, suffix:str='last', save_path:str='./checkpoints'):
    checkpoint = OrderedDict()
    checkpoint['module'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['scheduler'] = scheduler.state_dict() if scheduler else None
    checkpoint['best_loss'] = best_loss
    checkpoint['epoch'] = epoch
    checkpoint_file_path = os.path.join(save_path,f"{model.name}_{suffix}.pt")
    torch.save(checkpoint, checkpoint_file_path)

def load_checkpoint(model, optimizer, scheduler=None,
        save_path:str='./checkpoints',suffix:str='last') -> Tuple[int,float]:

    checkpoint_file_path = os.path.join(save_path,f"{model.name}_{suffix}.pt")
    assert os.path.isfile(checkpoint_file_path),f"checkpoint does not found for given directory {save_path}"
    state_dict = torch.load(checkpoint_file_path)
    assert "module" in state_dict,"module not found in the state dictionary"
    assert "optimizer" in state_dict,"optimizer not found in the state dictionary"
    assert "scheduler" in state_dict,"scheduler not found in the state dictionary"
    assert "best_loss" in state_dict,"best_loss not found in the state dictionary"
    assert "epoch" in state_dict,"epoch not found in the state dictionary"

    model.load_state_dict(state_dict['module'])
    try:
        optimizer.load_state_dict(state_dict['optimizer'])
    except Exception as e:
        print("Warning! optimizer is changed\n",e)
    if scheduler: scheduler.load_state_dict(state_dict['scheduler'])
    best_loss = state_dict['best_loss']
    epoch = state_dict['epoch']
    return best_loss,epoch

def load_model(model,save_path:str='./checkpoints',suffix:str='best'):
    checkpoint_file_path = os.path.join(save_path,f"{model.name}_{suffix}.pt")
    assert os.path.isfile(checkpoint_file_path),f"checkpoint does not found for given directory {save_path}"
    state_dict = torch.load(checkpoint_file_path)
    assert "module" in state_dict,"module not found in the state dictionary"
    assert "optimizer" in state_dict,"optimizer not found in the state dictionary"
    assert "scheduler" in state_dict,"scheduler not found in the state dictionary"
    assert "best_loss" in state_dict,"best_loss not found in the state dictionary"
    assert "epoch" in state_dict,"epoch not found in the state dictionary"

    model.load_state_dict(state_dict['module'])
