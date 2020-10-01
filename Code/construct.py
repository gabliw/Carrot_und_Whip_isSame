from __future__ import print_function
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', help='cifar10 | celebA | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--param', type=str, default='base', help='parameter json file load')

opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = datasets.ImageFolder(root='../Dataset/celeba',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

breakpoint()


# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))