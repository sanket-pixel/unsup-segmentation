import torch
import os
from torch.utils.data import DataLoader
from src.models.discriminator import Discriminator
from src.models.segmentor import UNet2D
from src.dataloder.scan_loader import ScanDataset
from src.models.dice_bce_loss import DiceBCELoss
import torch
import numpy as np
from tqdm import tqdm
import configparser
from matplotlib import pyplot as plt
from pathlib import Path
from einops import rearrange
import copy
import itertools
import pandas as pd
import gc

torch.autograd.set_detect_anomaly(True)

config = configparser.ConfigParser()
config.read(os.path.join("configs", "scan_seg.config"))
scan_path = config.get("Dataloader", "scan_path")
mask_path = config.get("Dataloader", "mask_path")
source_domain = config.get("Dataloader", "source_domain")
target_domain = config.get("Dataloader", "target_domain")
scan_type = config.get("Dataloader", "scan_type")
batch_size = config.getint("Dataloader", "batch_size")

scan_dataset_train = ScanDataset(source_domain, target_domain, scan_path, mask_path, scan_type, "training")
scan_dataset_eval = ScanDataset(source_domain, target_domain, scan_path, mask_path, scan_type, "validation")

dataloader_train = DataLoader(scan_dataset_train, batch_size=batch_size, shuffle=True)
dataloader_eval = DataLoader(scan_dataset_eval, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, stats, model_name):
    model_dict = {"model": model, "stats": stats}
    torch.save(model_dict, "../models/" + model_name + ".pth")


@torch.no_grad()
def eval_model(model):
    """ Computing model accuracy """
    correct = 0
    total = 0
    loss_list = []
    label_list = []

    criterion = torch.nn.BCELoss().to(device)  # cross entropy loss
    # model = model.eval()
    for batch in dataloader_eval:
        scans = batch[0].float().to(device)
        labels = batch[1].float().to(device)
        label_list.append(labels)

        # Forward pass only to get logits/output
        outputs = model(scans)  # forward pass
        predictions = torch.zeros_like(labels)
        predictions[torch.where(outputs >= 0.5)] = 1
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Get predictions from the maximum value
        correct += len(torch.where(labels == predictions)[0])
        total += len(labels)

    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    print(accuracy)
    return accuracy, loss


def train_model():
    LR = config.getfloat("Classification", "LR")
    EPOCHS = config.getint("Classification", "EPOCHS")
    EVAL_FREQ = config.getint("Classification", "EVAL_FREQ")
    SAVE_FREQ = config.getint("Classification", "SAVE_FREQ")
    model = config.get("Classification", "model")
    # initialize model
    # for using with optical flow change modality to "optical_flow"
    segmentor = UNet2D(n_chans_in=1, n_chans_out=1).to(device)
    segmentor_loss = DiceBCELoss().to(device)

    discriminator = Discriminator().to(device)
    discriminator_loss = torch.nn.BCELoss().to(device)  # cross entropy loss

    adversarial_loss = torch.nn.BCELoss().to(device)

    params = list(discriminator.parameters()) + list(segmentor.parameters())
    optimizer = torch.optim.Adam(params=params, lr=LR)  # define optimizer
    stats = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "accuracy": []
    }
    init_epoch = 0
    loss_hist = []
    for epoch in range(init_epoch, EPOCHS):  # iterate over epochs
        loss_list = []
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, batch in progress_bar:  # iterate over batches
            scans = batch[0].float().to(device)
            masks = batch[1].float().to(device)
            labels = batch[2].float().to(device)
            optimizer.zero_grad()  # remove old grads

            scans = rearrange(scans, "b s c h w -> (b s) c h w")
            masks = rearrange(masks, "b s c h w -> (b s) c h w")
            labels = rearrange(labels, "b s l->(b s) l")

            predicted_masks, x3 = segmentor(scans)  # get predictions
            s_loss = segmentor_loss(predicted_masks, masks)

            x3_detached = x3.clone().detach()
            predicted_labels = discriminator(x3_detached)
            d_loss = discriminator_loss(predicted_labels, labels)  # find loss

            total_loss = s_loss + d_loss
            loss_list.append(total_loss.item())

            total_loss.backward(retain_graph=True)
            optimizer.step()  # update weights
            del predicted_labels,predicted_masks,x3, x3_detached, d_loss,s_loss, total_loss
            gc.collect()



            # find indices of target domains
            discriminator_state_dict = copy.deepcopy(discriminator.state_dict())
            target_indices = torch.where(labels == 0)[0]
            if target_indices.shape[0] > 0:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                adversarial_labels = torch.logical_not(labels).float()[target_indices]
                adversarial_scans = scans[target_indices]
                mask_new, x3_new = segmentor(adversarial_scans)
                adversarial_predicted_labels = discriminator(x3_new)
                a_loss = adversarial_loss(adversarial_predicted_labels, adversarial_labels)
                a_loss.backward()  # find gradients
                optimizer.step()  # update weights
                discriminator.load_state_dict(discriminator_state_dict)
                del x3_new, mask_new, adversarial_predicted_labels,a_loss, discriminator_state_dict
                gc.collect()
        #     progress_bar.set_description(f"Epoch {0 + epoch} Iter {i + 1}: loss {loss.item():.5f}. ")
        # print("Average Loss", np.mean(loss_list))
        # # update stats
        # loss_hist.append(np.mean(loss_list))
        # stats['epoch'].append(epoch)
        # stats['train_loss'].append(loss_hist[-1])
        #
        # if epoch % EVAL_FREQ == 0:
        #     accuracy, valid_loss = eval_model(unet)
        #     print(f"Accuracy at epoch {epoch}: {round(accuracy, 2)}%")
        # else:
        #     accuracy, valid_loss = -1, -1
        # stats["accuracy"].append(accuracy)
        # stats["valid_loss"].append(valid_loss)
        # if epoch % SAVE_FREQ == 0:
        #     model_name = model + "_" + source_domain + "_" + target_domain + "_" + scan_type + ".pth"
        #     save_model(unet, stats, model_name)


# train_model()
# unet = torch.load("/home/sanket/Desktop/Projects/unsup-segmentation/models/Unet_without_adverserial.pth")
# model = unet['model'].eval()
# eval_model(model)

def plot_results(source_domain, target_domain):
    model = config.get("Classification", "model")
    model_name = model + "_" + source_domain + "_" + target_domain + "_" + scan_type + ".pth"
    model_dict = torch.load("../models/" + model_name + ".pth")
    plt.plot(model_dict["stats"]["train_loss"], label="Training Loss")
    plt.plot(model_dict["stats"]["valid_loss"], label="Validation Loss")
    plt.suptitle(source_domain + " : " + target_domain + " Loss", fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    folder_name = source_domain + "_" + target_domain
    folder_path = os.path.join("..", "figures", folder_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    loss_path = os.path.join(folder_path, "loss.jpg")
    plt.legend()
    plt.savefig(loss_path)
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(model_dict["stats"]["accuracy"], label="Accuracy")
    plt.suptitle(source_domain + " : " + target_domain + " Accuracy", fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    accuracy_path = os.path.join(folder_path, "accuracy.jpg")
    plt.legend()
    plt.savefig(accuracy_path)
    plt.clf()
    plt.cla()
    plt.close()


def show_accuracy_table(source_domain, target_domain, accuracy_dict):
    model = config.get("Classification", "model")
    model_name = model + "_" + source_domain + "_" + target_domain + "_" + scan_type + ".pth"
    model_dict = torch.load("../models/" + model_name + ".pth")
    accuracy_dict[source_domain + "_" + target_domain] = [model_dict["stats"]["accuracy"][-1]]


# combinations = [["siemens", "ge"], ["siemens", "philips"],
#                 ["ge", "siemens"], ["ge", "philips"],
#                 ["philips", "ge"], ["philips", "siemens"]]
# accuracy_dict = {}
# for c in combinations:
#     show_accuracy_table(c[0], c[1],accuracy_dict)
# print(pd.DataFrame(accuracy_dict).transpose())

train_model()
