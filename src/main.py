import torch
import os
from torch.utils.data import DataLoader
from src.models.discriminator import Discriminator
from src.models.segmentor import UNet2D
from src.dataloder.scan_loader import ScanDataset
from src.models.dice_bce_loss import DiceBCELoss, DiceScore
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
batch_size_train = config.getint("Dataloader", "batch_size_train")
batch_size_eval = config.getint("Dataloader", "batch_size_eval")

scan_dataset_train = ScanDataset(source_domain, target_domain, scan_path, mask_path, scan_type, "training")
scan_dataset_eval = ScanDataset(source_domain, target_domain, scan_path, mask_path, scan_type, "training")

dataloader_train = DataLoader(scan_dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=4)
dataloader_eval = DataLoader(scan_dataset_eval, batch_size=batch_size_train, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, stats, model_name):
    model_dict = {"model": model, "stats": stats}
    torch.save(model_dict, "../models/" + model_name + ".pth")


@torch.no_grad()
def eval_model(segmentor, discriminator):
    """ Computing model accuracy """
    correct = 0
    total = 0
    d_loss_list = []
    s_loss_list = []
    label_list = []
    dice_score_list = []

    segmentor_loss = DiceBCELoss().to(device)
    discriminator_loss = torch.nn.BCELoss().to(device)
    dice_score_func = DiceScore().to(device)

    # model = model.eval()
    for batch in dataloader_eval:
        scans = batch[0].float().to(device)
        masks = batch[1].float().to(device)
        labels = batch[2].float().to(device)

        target_indices = torch.where(labels == 0)[0]

        scans = rearrange(scans, "b s c h w -> (b s) c h w")
        masks = rearrange(masks, "b s c h w -> (b s) c h w")
        labels = rearrange(labels, "b s l->(b s) l")
        scans_target = scans[target_indices]
        masks_target = masks[target_indices]
        labels_target = labels[target_indices]
        predicted_mask, x3 = segmentor(scans_target)
        predicted_mask[predicted_mask <= 0.5] = 0
        predicted_mask[predicted_mask > 0.5] = 1
        predicted_label = discriminator(x3)
        prediction = torch.zeros_like(labels_target)
        prediction[torch.where(predicted_label >= 0.5)] = 1
        d_loss = discriminator_loss(predicted_label, labels_target)
        d_loss_list.append(d_loss.item())
        # get dice score
        dice_score = dice_score_func(predicted_mask, masks_target)
        dice_score_list.append(dice_score.item())
        s_loss = segmentor_loss(predicted_mask, masks_target)
        s_loss_list.append(s_loss.item())
        # Get predictions from the maximum value
        correct += len(torch.where(labels_target == prediction)[0])
        total += len(labels_target)



        # # Forward pass only to get logits/output
        # for i, scan in enumerate(scans):
        #     label = labels[i]
        #     mask = masks[i]
        #     scan = scan.unsqueeze(0)
        #     mask = mask.unsqueeze(0)
        #     label = label.unsqueeze(0)
        #     predicted_mask, x3 = segmentor(scan)
        #     predicted_mask[predicted_mask <= 0.5] = 0
        #     predicted_mask[predicted_mask > 0.5] = 1
        #     predicted_label = discriminator(x3)
        #     prediction = torch.zeros_like(label)
        #     prediction[torch.where(predicted_label >= 0.5)] = 1
        #     d_loss = discriminator_loss(predicted_label, label)
        #     d_loss_list.append(d_loss.item())
        #     # get dice score
        #     dice_score = dice_score_func(predicted_mask, mask)
        #     dice_score_list.append(dice_score.item())
        #     s_loss = segmentor_loss(predicted_mask, mask)
        #     s_loss_list.append(s_loss.item())
        #
        #     # Get predictions from the maximum value
        #     correct += len(torch.where(label == prediction)[0])
        #     total += len(label)

    # Total correct predictions and loss
    accuracy = correct / total * 100
    d_loss_mean = np.mean(d_loss_list)
    s_loss_mean = np.mean(s_loss_list)
    dice_score_mean = np.mean(dice_score_list)

    return accuracy, dice_score_mean, d_loss_mean, s_loss_mean


def train_model():
    LR_a = config.getfloat("Classification", "LR_a")
    LR_d = config.getfloat("Classification", "LR_d")
    LR_s = config.getfloat("Classification", "LR_s")
    EPOCHS = config.getint("Classification", "EPOCHS")
    EVAL_FREQ = config.getint("Classification", "EVAL_FREQ")
    SAVE_FREQ = config.getint("Classification", "SAVE_FREQ")
    model = config.get("Classification", "model")
    segmentor_path = os.path.join("..","models","ge3_updateddata_train70_25epochs_combinedloss.pkl")
    # initialize model
    # for using with optical flow change modality to "optical_flow"
    segmentor = torch.load(segmentor_path)
    discriminator = Discriminator().to(device)

    segmentor_loss = DiceBCELoss().to(device)
    discriminator_loss = torch.nn.BCELoss().to(device)  # cross entropy loss
    adversarial_loss = torch.nn.BCELoss().to(device)

    # params = list(discriminator.parameters()) + list(segmentor.parameters())
    optimizer_a = torch.optim.Adam(params=segmentor.parameters(), lr=LR_a)  # define optimizer
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=LR_d)
    optimizer_s = torch.optim.Adam(params=segmentor.parameters(), lr=LR_s)  # define optimizer
    stats = {
        "epoch": [],
        "train_loss": [],
        "valid_d_loss": [],
        "valid_s_loss":[],
        "accuracy": [],
        "a_loss": [],
        "s_loss": [],
        "d_loss": [],
        "target_dice_score" :[]
    }
    init_epoch = 0
    a_loss_hist = []
    d_loss_hist = []
    s_loss_hist = []
    for epoch in range(init_epoch, EPOCHS):  # iterate over epochs
        a_loss_list, d_loss_list, s_loss_list = [], [], []
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, batch in progress_bar:  # iterate over batches
            scans = batch[0].float().to(device)
            masks = batch[1].float().to(device)
            labels = batch[2].float().to(device)

            target_indices = torch.where(labels == 0)[0]
            source_indices = torch.where(labels == 1)[0]

            optimizer_d.zero_grad(set_to_none=True)  # remove old grads
            optimizer_s.zero_grad(set_to_none=True)
            optimizer_a.zero_grad(set_to_none=True)

            scans = rearrange(scans, "b s c h w -> (b s) c h w")
            masks = rearrange(masks, "b s c h w -> (b s) c h w")
            labels = rearrange(labels, "b s l->(b s) l")

            # get segmentor loss only for source domains
            predicted_masks, x3 = segmentor(scans)  # get predictions
            if target_indices.shape[0] > 0:
                predicted_masks_source = predicted_masks[source_indices]
                mask_source = masks[source_indices]
                s_loss = segmentor_loss(predicted_masks_source, mask_source)
            else:
                s_loss = torch.Tensor([0]).to(device)
                s_loss.requires_grad = True

            # get discriminator loss for source and target
            x3_detached = x3.clone().detach()
            predicted_labels = discriminator(x3_detached)
            d_loss = discriminator_loss(predicted_labels, labels)  # find loss

            # update discriminator with discriminator loss
            d_loss.backward(retain_graph=True)
            optimizer_d.step()

            # get adversarial loss only for target domain
            if target_indices.shape[0] > 0:
                adversarial_x3 = x3[target_indices]
                adversarial_labels = torch.logical_not(labels).float()[target_indices]
                adversarial_predicted_labels = discriminator(adversarial_x3)
                a_loss = adversarial_loss(adversarial_predicted_labels, adversarial_labels)
            else:
                a_loss = torch.Tensor([0]).to(device)
                a_loss.requires_grad = True

            # update segementor with adversarial and segementor loss
            a_loss.backward(retain_graph=True)
            s_loss.backward()
            optimizer_s.step()  # update weights
            optimizer_a.step()

            # update loss lists
            a_loss_list.append(a_loss.item())
            d_loss_list.append(d_loss.item())
            s_loss_list.append(s_loss.item())
            gc.collect()
            torch.cuda.empty_cache()
            progress_bar.set_description(f"Epoch {0 + epoch} Iter {i + 1}: adv loss {a_loss.item():.5f}. ")
            del predicted_labels, predicted_masks, x3, d_loss, s_loss, a_loss

        print("Average Adversarial Loss", np.mean(a_loss_list))
        print("Average Segmentation Loss", np.mean(s_loss_list))
        print("Average Discrimination Loss", np.mean(d_loss_list))
        # update stats
        a_loss_hist.append(np.mean(a_loss_list))
        d_loss_hist.append(np.mean(d_loss_list))
        s_loss_hist.append(np.mean(s_loss_list))

        stats['epoch'].append(epoch)
        stats['a_loss'].append(a_loss_hist[-1])
        stats['s_loss'].append(s_loss_hist[-1])
        stats['d_loss'].append(d_loss_hist[-1])

        if epoch % EVAL_FREQ == 0:
            accuracy, dice_score_mean, d_loss_mean, s_loss_mean = eval_model(segmentor, discriminator)
            stats["target_dice_score"].append(dice_score_mean*100)
            stats["accuracy"].append(accuracy)
            stats["valid_d_loss"].append(d_loss_mean)
            stats["valid_s_loss"].append(s_loss_mean)
            print(f"Accuracy at epoch {epoch}: {round(accuracy, 2)}%")
            print(f"Dice Score at epoch {epoch}: {round(dice_score_mean*100, 2)}%")

        if epoch % SAVE_FREQ == 0:
            model_name = model + "_" + source_domain + "_" + target_domain + "_" + scan_type + ".pth"
            save_model(discriminator, stats, model_name)


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


# def show_accuracy_table(source_domain, target_domain, accuracy_dict):
#     model = config.get("Classification", "model")
#     model_name = model + "_" + source_domain + "_" + target_domain + "_" + scan_type + ".pth"
#     model_dict = torch.load("../models/" + model_name + ".pth")
#     accuracy_dict[source_domain + "_" + target_domain] = [model_dict["stats"]["accuracy"][-1]]


# combinations = [["siemens", "ge"], ["siemens", "philips"],
#                 ["ge", "siemens"], ["ge", "philips"],
#                 ["philips", "ge"], ["philips", "siemens"]]
# accuracy_dict = {}
# for c in combinations:
#     show_accuracy_table(c[0], c[1],accuracy_dict)
# print(pd.DataFrame(accuracy_dict).transpose())

train_model()
