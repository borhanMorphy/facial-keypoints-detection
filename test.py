from transforms.train import TrainTransforms

if __name__ == "__main__":
    from dataset import FKDataset_train
    ds = FKDataset_train('./data/training_fixed',
        transforms=TrainTransforms(96,96),
        transform=None,
        target_transform=None)


    for img,targets in ds:
        pass