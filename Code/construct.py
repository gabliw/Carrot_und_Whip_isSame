from __future__ import print_function
import os
import random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision.datasets as dest
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--param', type=str, default='base', help='parameter json file load')

opt = parser.parse_args()
print(opt)