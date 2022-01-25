import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from pan.networks import PAN, ResNet50, Mask_Classifier

true_masks = torch.randn(2,512,767)
pred_masks = torch.randn(2,256,128,192)

criterion = nn.CrossEntropyLoss()
mask_classifier = Mask_Classifier(256,2)

pred_masks = mask_classifier(pred_masks)
true_masks = F.interpolate(true_masks, scale_factor=0.4,recompute_scale_factor=True)
true_masks = true_masks.long().squeeze(1)

loss = criterion(pred_masks,true_masks) + dice_loss(F.softmax(pred_mask,dim=1).float()
                                                    ,F.one_hot(true_masks,2).permute(0,3,1,2).float())

print('loss : {}'.format(loss))