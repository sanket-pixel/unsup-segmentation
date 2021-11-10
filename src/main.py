import torch
import os
from torch.utils.data import DataLoader
from src.models.unet import UNet2D
from src.dataloder.scan_loader import ScanDataset
import torch
import numpy as np
from tqdm import tqdm



source_domain = 'philips'
target_domain = 'siemens'
scan_type = '3'

scan_dataset = ScanDataset(source_domain, target_domain, scan_type)

scan_dataloader = DataLoader(scan_dataset, batch_size=4, shuffle=True)


def train_model():
    LR = 1e-4 # learning rate
    EPOCHS = 20
    # initialize model
    # for using with optical flow change modality to "optical_flow"
    unet = UNet2D(n_chans_in=1, n_chans_out=2, mode="feature_discriminator")
    criterion = torch.nn.CrossEntropyLoss() # cross entropy loss
    optimizer = torch.optim.Adam(params=unet.parameters(), lr = LR) # define optimizer
    stats = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "accuracy": []
    }
    init_epoch = 0
    loss_hist = []
    for epoch in range(init_epoch, EPOCHS): # iterate over epochs
        loss_list = []
        progress_bar = tqdm(enumerate(scan_dataloader), total=len(scan_dataloader))
        for i, batch in progress_bar: # iterate over batches
            scans = batch[0]
            labels = batch[1]
            optimizer.zero_grad() # remove old grads
            y = unet(scans) # get predictions
            loss = criterion(y, labels) # find loss
            loss_list.append(loss.item())
            loss.backward() # find gradients
            optimizer.step() # update weights
            progress_bar.set_description(f"Epoch {0 + 1} Iter {i + 1}: loss {loss.item():.5f}. ")
        # update stats
        loss_hist.append(np.mean(loss_list))
        stats['epoch'].append(epoch)
        stats['train_loss'].append(loss_hist[-1])


train_model()