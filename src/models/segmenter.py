import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time, os, copy
from PIL import Image
import natsort
import cv2
import random 
import torchvision.transforms.functional as TF
import os
import cv2
from tqdm import tqdm
import nibabel as nib
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from statistics import mean


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


###################################################################################################################################################################################
#### Loss
##################################################################################################################################################################################


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth= 0.0000001):
      
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()  
                                 
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE, BCE, dice_loss

###################################################################################################################################################################################
#### Dataloader 
###################################################################################################################################################################################

class ScanDataset(Dataset):
    def __init__(self, mode = 'train'):
      
      self.original_images = np.load('/content/gdrive/MyDrive/GE/ge3_'+ mode + '_original_images.npy')
      self.mask_images = np.load('/content/gdrive/MyDrive/GE/ge3_'+ mode + '_mask_images.npy')

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):

        original_batch = torch.from_numpy(self.original_images[idx]).float()
        mask_batch = torch.from_numpy(self.mask_images[idx]).float()

        return original_batch, mask_batch


###################################################################################################################################################################################
#### MODEL 
###################################################################################################################################################################################

from torch import nn
from dpipe.layers.resblock import ResBlock2d
from dpipe.layers.conv import PreActivation2d


class UNet2D(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, kernel_size=3, padding=1, pooling_size=2, n_filters_init=8,
                 dropout=False, p=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.pooling_size = pooling_size
        n = n_filters_init
        if dropout:
            dropout_layer = nn.Dropout(p)
        else:
            dropout_layer = nn.Identity()

        self.init_path = nn.Sequential(
            nn.Conv2d(n_chans_in, n, self.kernel_size, padding=self.padding, bias=False),
            nn.ReLU(),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n, n, kernel_size=self.kernel_size, padding=self.padding)
        )
        self.shortcut0 = nn.Conv2d(n, n, 1)

        self.down1 = nn.Sequential(
            nn.BatchNorm2d(n),
            nn.Conv2d(n, n * 2, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding)
        )
        self.shortcut1 = nn.Conv2d(n * 2, n * 2, 1)

        self.down2 = nn.Sequential(
            nn.BatchNorm2d(n * 2),
            nn.Conv2d(n * 2, n * 4, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding)
        )
        self.shortcut2 = nn.Conv2d(n * 4, n * 4, 1)

        self.down3 = nn.Sequential(
            nn.BatchNorm2d(n * 4),
            nn.Conv2d(n * 4, n * 8, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            dropout_layer
        )

        self.up3 = nn.Sequential(
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 8, n * 8, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 8),
            nn.ConvTranspose2d(n * 8, n * 4, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.up2 = nn.Sequential(
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 4, n * 4, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 4),
            nn.ConvTranspose2d(n * 4, n * 2, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.up1 = nn.Sequential(
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            ResBlock2d(n * 2, n * 2, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm2d(n * 2),
            nn.ConvTranspose2d(n * 2, n, kernel_size=self.pooling_size, stride=self.pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.out_path = nn.Sequential(
            ResBlock2d(n, n, kernel_size=1),
            PreActivation2d(n, n_chans_out, kernel_size=1),
            nn.BatchNorm2d(n_chans_out)
        )

    def forward(self, x):
        x0 = self.init_path(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x2_up = self.up3(x3)
        x1_up = self.up2(x2_up + self.shortcut2(x2))
        x0_up = self.up1(x1_up + self.shortcut1(x1))
        x_out = self.out_path(x0_up + self.shortcut0(x0))

        return F.sigmoid(x_out)

###################################################################################################################################################################################
#### PLOTTING 
###################################################################################################################################################################################


def plotting(path, current_epoch, batch_idx, original, mask, pred):
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
  fig.set_size_inches(6, 8, forward=True)
  fig.set_dpi(120)
  fig.tight_layout()
  ax1.imshow(original[0][0].detach().cpu().numpy(), cmap = 'gray')
  ax2.imshow(mask[0][0].detach().cpu().numpy(), cmap = 'gray')
  ax3.imshow(pred[0][0].detach().cpu().numpy(), cmap = 'gray')
  ax1.set_title('original')
  ax2.set_title('mask')
  ax3.set_title('pred')
  plt.savefig(path + str(current_epoch) + '_' + str(batch_idx))
  plt.pause(0.00001)

###################################################################################################################################################################################
#### TRAINING
###################################################################################################################################################################################



batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True ,num_workers = 0)

val_batch_size = 16
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size = val_batch_size, shuffle = True ,num_workers = 0)


def train(model, train_loader, optimizer, loss_func, current_epoch, device):
  
  # train 
  model.train()

  loss_hist = []  # loss
  BCE_loss_hist = []
  dice_loss_hist = []

  for batch_idx, data in enumerate(train_loader):

      optimizer.zero_grad()

      original = torch.unsqueeze(data[0].float(), axis = 1).to(device)
      
      mask = torch.unsqueeze(data[1].float(), axis = 1).to(device)
          
      pred = model(original)

      loss, BCE, dice_loss = loss_func(pred, mask)

      if batch_idx %  10 == 0:
        print( "Train: Epoch",current_epoch,"| Batch:",batch_idx,"| Combined Loss:", loss.item(),"| BCE",BCE.item(),"| dice_loss",dice_loss.item() )

      if batch_idx % 100 == 0:
        plotting('/content/gdrive/MyDrive/VMIA_Lab_Data/Plots/DART20/Combined_Loss/Train/plot_',current_epoch, batch_idx, original, mask, pred)

      loss_hist.append(loss.item())
      BCE_loss_hist.append(BCE.item())
      dice_loss_hist.append(dice_loss.item())

      loss.backward()

      optimizer.step()

  return model, optimizer, loss_hist, BCE_loss_hist, dice_loss_hist


model = UNet2D(1,1).to(device)

optimizer = optim.Adam( model.parameters(), lr = 0.001 )

loss_func = DiceBCELoss()


train_loss_dict = {}
val_loss_dict = {}
train_loss_means = []
val_loss_means = []

train_BCE_loss_means = []
val_BCE_loss_means = []
train_dice_loss_means = []
val_dice_loss_means = []

for epoch in range(no_of_epochs):

  model, optimizer, loss_hist, BCE_loss_hist, dice_loss_hist = train(model, train_loader, optimizer, loss_func, epoch, device)

  print("Avg train loss for epoch  -->", mean(loss_hist),"| BCE:",mean(BCE_loss_hist),"| dice_loss:",mean(dice_loss_hist))
  train_loss_means.append(mean(loss_hist))
  train_BCE_loss_means.append(mean(BCE_loss_hist))
  train_dice_loss_means.append(mean(dice_loss_hist))

  train_loss_dict[epoch] = loss_hist

  val_loss_hist, val_BCE_loss_hist, val_dice_loss_hist = val(model, val_loader, loss_func, epoch, device)

  print("Average val loss for epoch  -->", mean(val_loss_hist),"| BCE:",mean(val_BCE_loss_hist),"| dice_loss:",mean(val_dice_loss_hist))

  val_loss_means.append(mean(val_loss_hist))
  val_BCE_loss_means.append(mean(val_BCE_loss_hist))
  val_dice_loss_means.append(mean(val_dice_loss_hist))

  val_loss_dict[epoch] = val_loss_hist

  print("****************************************************************************************************")