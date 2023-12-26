
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image
import torch
from glob import glob
import os
import clip
import torch
import torch.nn.functional as F

current_path = "C:\\Users\\Leyan\\OneDrive\\Traffic Sign Dataset"
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class traffic_Dataset(Dataset):
    def __init__(self, root, split, transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform

        self.imgs = glob(f'{root}*/*')
        self.lbls = [int(os.path.basename(os.path.dirname(path))) for path in self.imgs]
        self.prompts = list(pd.read_csv(os.path.join(current_path, "distinct_labels.csv")).Name)
        assert len(self.imgs) == len(self.lbls), 'mismatched length!'
        print('Total data in {} split: {}'.format(split, len(self.imgs)))


    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath = self.imgs[index]
        img = Image.open(imgpath).convert('RGB')
        lbl = int(self.lbls[index])
        prompt_text = "This is " + self.prompts[lbl]
        prompt = clip.tokenize(prompt_text)
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl, prompt

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgs)


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y, z = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y, z

    def __len__(self):
        return len(self.subset)


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        # focal_weights = [0.6092928112215079, 1.7974137931034482, 0.8987068965517241, 0.276525198938992, 0.7336382828993666, 0.3706007820831852, 0.9217506631299734, 0.47300362976406535, 8.987068965517242, 35.94827586206897, 1.0270935960591132, 0.5209895052473763, 0.7489224137931034, 1.9971264367816093, 0.5616918103448276, 3.268025078369906, 0.5063137445361826, 0.553050397877984, 8.987068965517242, 17.974137931034484, 3.9942528735632186, 5.991379310344827, 3.9942528735632186, 5.135467980295567, 0.7189655172413794, 35.94827586206897, 0.5706075533661741, 2.5677339901477834, 0.16120303077160972, 1.634012539184953, 0.4793103448275862, 1.7118226600985222, 5.135467980295567, 17.974137931034484, 2.7652519893899203, 0.4608753315649867, 1.7974137931034482, 1.2395957193816884, 2.396551724137931, 2.114604462474645, 2.2467672413793105, 3.9942528735632186, 2.2467672413793105, 0.8767872161480236, 2.396551724137931, 2.9956896551724137, 3.9942528735632186, 5.991379310344827, 7.189655172413793, 1.7118226600985222, 1.2838669950738917, 8.987068965517242, 1.9971264367816093, 35.94827586206897, 0.2219029374201788, 0.4438058748403576, 0.6536050156739812, 11.982758620689655]
        focal_weights = torch.ones((64))
        focal_weights = torch.FloatTensor(focal_weights)
        self.alpha = torch.FloatTensor(focal_weights).half().cuda(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss