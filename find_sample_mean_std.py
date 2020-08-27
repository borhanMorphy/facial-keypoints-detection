
from dataset import FKDataset_train
import torchvision.transforms as vis_transforms
import torch
from tqdm import tqdm

def main(root_path:str):
    transform = vis_transforms.ToTensor()
    ds = FKDataset_train(root_path=root_path,transform=transform)
    sample = torch.empty(len(ds), dtype=torch.float32)
    for i,(img,_) in tqdm(enumerate(ds), total=len(ds)):
        sample[i] = img.mean()

    mean = sample.mean()
    std = sample.std()
    print(f"mean: {mean}\tstd: {std}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])