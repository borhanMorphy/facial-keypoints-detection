from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
from cv2 import cv2
from random import shuffle
from typing import List
from utils import load_img

class FKDataset_train(Dataset):
    __label_mapper__ = [
        'left_eye_center_x', 'left_eye_center_y',
        'right_eye_center_x', 'right_eye_center_y',
        'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
        'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
        'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
        'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
        'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
        'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
        'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        'nose_tip_x', 'nose_tip_y',
        'mouth_left_corner_x', 'mouth_left_corner_y',
        'mouth_right_corner_x', 'mouth_right_corner_y',
        'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']

    def __init__(self,root_path:str, img_ids, label_mapper:List=None,
            transforms=None, transform=None, target_transform=None):
        self.ids = []
        self.targets = []
        self.label_mapper = label_mapper if label_mapper else FKDataset_train.__label_mapper__

        for name in img_ids:
            targets = []
            with open(os.path.join(root_path,name+".txt"),"r") as foo:
                data = foo.read().split("\n")
                for row in data:
                    assert len(row.split(" ")) == 2,f"broken data {row}"
                    label,value = row.split(" ")
                    if value == 'None':
                        targets = []
                        break
                    targets.append(float(value))
            if len(targets) == 0:
                continue

            self.targets.append(np.array(targets, dtype=np.float32))
            self.ids.append(os.path.join(root_path, name+".jpg"))

        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,idx:int):
        img = load_img(self.ids[idx])
        targets = self.targets[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            targets = self.target_transform(targets)
        if self.transforms:
            img,targets = self.transforms(img,targets)

        return img,targets

    def __len__(self):
        return len(self.ids)

def get_training_datasets(root_path:str, train_transforms, val_transforms, split_ratios):
    assert sum(split_ratios) == 1.0
    names = [os.path.splitext(fname)[0] for fname in os.listdir(root_path) if fname.endswith(".jpg")]
    shuffle(names)
    current = 0
    datasets = []
    for i,split_ratio in enumerate(split_ratios):
        selection_count = int(split_ratio*len(names))
        subnames = names[current:selection_count+current]
        current += selection_count
        if i == 0:
            transforms,transform,target_transform = train_transforms
        else:
            transforms,transform,target_transform = val_transforms
        datasets.append(
            FKDataset_train(root_path, subnames,
                transforms=transforms,
                transform=transform,
                target_transform=target_transform) )

    return datasets


if __name__ == "__main__":
    import sys
    ds = get_training_datasets(sys.argv[1])[0]

    for img,targets in ds:

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        for x,y in targets.reshape(-1,2):
            cv2.circle(img, (int(x),int(y)), 1, (0,0,255))
        cv2.imshow("",img)
        if cv2.waitKey(0) == 27:
            break