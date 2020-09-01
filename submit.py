import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import get_backbone, get_available_backbones
from model import FacialKeypointsDetector

from dataset import get_test_dataset,labels

import os
import argparse
import json
from typing import Dict

from cv2 import cv2
import numpy as np
import transforms
import math
from utils import load_model
import pandas as pd
from tqdm import tqdm
# TODO write docstring

def parse_json(file_path:str) -> Dict:
    with open(file_path,"r") as foo:
        data = json.load(foo)
    return data

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-data-path', '-tdp', type=str, required=True)
    ap.add_argument('--submission-file-path', '-sfp', type=str, required=True)
    ap.add_argument('--idlookup-file-path', '-ifp', type=str, required=True)
    ap.add_argument('--checkpoint-path', '-ckpt', type=str, default='./checkpoints')

    ap.add_argument('--backbone', '-b',type=str, choices=get_available_backbones())

    ap.add_argument('--num-classes', '-nc', type=int, default=30)
    ap.add_argument('--device', '-d', type=str, choices=['cpu','cuda'], default='cuda')

    ap.add_argument('--batch-size', '-bs', type=int, default=64)

    return vars(ap.parse_args())

def custom_collate_fn(batch):
    imgs,img_ids = zip(*batch)
    imgs = torch.stack(imgs,dim=0)
    return imgs,img_ids

def main(**kwargs):
    test_path = kwargs.get('test_data_path')
    submission_file_path = kwargs.get('submission_file_path')
    idlookup_file_path = kwargs.get('idlookup_file_path')
    checkpoint_path = kwargs.get('checkpoint_path')

    backbone_name = kwargs.get('backbone')

    num_classes = kwargs.get('num_classes')
    device = kwargs.get('device')

    batch_size = kwargs.get('batch_size')
    
    # TODO give trained mean and std
    mean = 0
    std = 1

    original_img_size = 96
    submission_df = pd.read_csv(submission_file_path)
    idlookup_df = pd.read_csv(idlookup_file_path)

    backbone = get_backbone(backbone_name)

    model = FacialKeypointsDetector(backbone, device=device, num_classes=num_classes)


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(model.get_input_size()),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)])

    dataset = get_test_dataset(root_path=test_path,transform=transform)

    dl = torch.utils.data.DataLoader(dataset,
        num_workers=4, batch_size=batch_size,
        pin_memory=True, collate_fn=custom_collate_fn)

    load_model(model,checkpoint_path)

    predictions = {}
    for imgs,img_ids in tqdm(dl, total=len(dl.dataset)//batch_size):
        # {img_id:{'loc1':pred}}
        preds = model.predict(imgs) * original_img_size
        for img_idx,pred in zip(img_ids,preds.cpu().numpy().tolist()):
            predictions[img_idx] = {}
            for label,location in zip(labels,pred):
                predictions[img_idx][label] = location

    locations = []
    row_ids = []
    for s,lookup in zip(submission_df.iterrows(), idlookup_df.iterrows()):
        s = s[1]
        lookup = lookup[1]
        # RowId,Location
        s = json.loads(s.to_json())
        # RowId,ImageId,FeatureName,Location
        lookup = json.loads(lookup.to_json())
        img_idx = lookup['ImageId']
        feature = lookup['FeatureName']
        row_id = s['RowId']
        location = predictions[img_idx][feature]
        locations.append(location)
        row_ids.append(row_id)
    new_submission_df = pd.DataFrame(data={'RowId':row_ids,'Location':locations})
    submissin_dir_path = os.path.dirname(submission_file_path)
    new_submission_file_path = os.path.join(submissin_dir_path,'submission.csv')
    print(new_submission_df.head())
    new_submission_df.to_csv(new_submission_file_path, index=False, header=1)

if __name__ == "__main__":
    kwargs = parse_arguments()
    print(kwargs)
    main(**kwargs)