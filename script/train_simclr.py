# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import os
import torch
from torch import nn
import random
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
# from   test.test_acc import test_acc
from   torch.utils.data import Dataset, DataLoader
from lightly.models.modules import NNMemoryBankModule
import timm



class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    data_transforms = transforms.Compose([  transforms.ToTensor(), 
                                            transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Lambda(lambda x: x * 2. - 1.),
                                            ])
    return data_transforms
   

class Dataset(Dataset):
    def __init__(self, root, train_class):
        datasets = np.load(root + "pokemon.npz")
        imgs = datasets['images']
        factors = datasets['labels']
        self.train_imgs = []
        self.train_factors = []
        for i in range(len(imgs)):
            self.train_imgs.append(np.squeeze(imgs[i]))
            self.train_factors.append(factors[i])
        del imgs
        del factors
        self.tranforms =  ContrastiveLearningViewGenerator(
                            get_simclr_pipeline_transform(64))
        
    def __getitem__(self, index):
        return self.tranforms(self.train_imgs[index]), self.train_factors[index]
    
    def __len__(self):
        return len(self.train_imgs)





class SIMCLR(nn.Module):
    def __init__(self):
        super().__init__()
        # model_fun, dim_in = model_dict[name]
        self.backbone = timm.create_model('tf_efficientnet_lite0', pretrained=True)
        self.projection_head = SimCLRProjectionHead(1000, 512, 128)

    def forward(self, x):
        f = self.backbone(x) 
        z = self.projection_head(f)
        return z


model = SIMCLR()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
memory_bank = NNMemoryBankModule(size=4096)
memory_bank.to(device)

trainset = Dataset(root='./', train_class=1)
dataset = LightlyDataset.from_torch_dataset(trainset)


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=[0.5, 0.99])

save_path = './Result/{}/'.format(random.randint(0, 10000))
model_dir = {"checkpoint":"./checkpoint", "samples":"./samples"}
for dir_ in model_dir:
    if not os.path.exists(save_path+model_dir[dir_]):
        os.makedirs(save_path+model_dir[dir_])

print("Starting Training")
for epoch in range(600):
    total_loss = 0
    for (data, _, _) in dataloader:
        x0 = data[0].to(device)
        x1 = data[1].to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    torch.save(model.state_dict(), "./C.pth")
    model.train()
