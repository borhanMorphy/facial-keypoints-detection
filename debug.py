from transforms.train import TrainTransforms
import numpy as np
from cv2 import cv2
from dataset import FKDataset_train,labels

if __name__ == "__main__":
    packed_labels = []
    for label in labels:
        l_name = "_".join(label.split("_")[:-1])
        if l_name not in packed_labels: packed_labels.append(l_name)

    ds = FKDataset_train('./data/training_fixed',
        transforms=TrainTransforms(224,96,debug=True,rotation_degree=90),
        transform=None,
        target_transform=None)

    for img,targets in ds:
        targets = targets.reshape(-1,2).astype(np.int32)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        for i,(target_x,target_y) in enumerate(targets):
            print("showing label: ",packed_labels[i])
            cimg = img.copy()
            cimg = cv2.circle(cimg,(target_x,target_y),5,(0,0,255))
            cv2.imshow("",cimg)
            if cv2.waitKey(0) == 27: exit(1)