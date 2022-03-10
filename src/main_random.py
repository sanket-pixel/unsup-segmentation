import torch
import os
from torch.utils.data import DataLoader
from src.models.discriminator import Discriminator
from src.models.segmentor import UNet2D
from shutil import copy
from src.dataloder.scan_loader import ScanDataset
from src.models.dice_bce_loss import DiceBCELoss, DiceScore
import torch
import numpy as np
from tqdm import tqdm
import configparser
from matplotlib import pyplot as plt
from pathlib import Path
import gc
import pandas as pd
import surface_distance.metrics as surf_dst
import cv2
import json

torch.autograd.set_detect_anomaly(True)

config = configparser.ConfigParser()
config.read(os.path.join("src", "configs", "experiment.config"))
scan_path = config.get("Dataloader", "scan_path")
mask_path = config.get("Dataloader", "mask_path")
source_domain = config.get("Dataloader", "source_domain")
target_domain = config.get("Dataloader", "target_domain")
scan_type = config.get("Dataloader", "scan_type")
batch_size_train = config.getint("Dataloader", "batch_size_train")
batch_size_eval = config.getint("Dataloader", "batch_size_eval")

scan_dataset_train = ScanDataset("training")
dataloader_train = DataLoader(scan_dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=4)

# dataloader_train = DataLoader(scan_dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=16)
# dataloader_eval = DataLoader(scan_dataset_eval, batch_size=batch_size_train, shuffle=False, num_workers=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_config():
    num_exp = len(os.listdir(os.path.join("results")))
    exp_id = num_exp + 1
    exp_folder = os.path.join("results", "experiment_" + str(exp_id))
    Path(exp_folder).mkdir(parents=True, exist_ok=True)
    config_path = os.path.join("src", "configs", "experiment.config")
    copy(config_path, exp_folder)
    return "experiment_" + str(exp_id)


def save_model(model, stats, exp):
    model_dict = {"model": model, "stats": stats}
    torch.save(model_dict, "models/" + exp + ".pth")


def save_sdat(sdat_scanwise, epoch, exp):
    sdat_folder = os.path.join("results", exp, "figures",
                               "validation_scans", "epoch" + "_" + str(epoch))
    sdat_filename = "sdat_scanwise" + "_" + str(epoch) + ".json"
    sdat_path = os.path.join(sdat_folder, sdat_filename)
    with open(sdat_path, 'w') as fp:
        json.dump(sdat_scanwise, fp)


def save_scan(scan_list, mask_list, predicted_mask_list, scan_id, epoch, exp):
    slice_id = 0
    for i, scan in enumerate(scan_list):
        mask = mask_list[i]
        predicted_mask = predicted_mask_list[i]
        scan = scan.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)
        predicted_mask = predicted_mask.permute(0, 2, 3, 1)
        for j, slice in enumerate(scan):
            scan_name = scan_id + "_" + str(slice_id) + ".jpeg"
            scan_folder = os.path.join("results", exp, "figures",
                                       "validation_scans", "epoch" + "_" + str(epoch), scan_id)
            Path(scan_folder).mkdir(parents=True, exist_ok=True)
            scan_path = os.path.join(scan_folder, scan_name)
            scan_s = slice.cpu().numpy()
            mask_s = mask[j].cpu().numpy()
            predicted_mask_s = predicted_mask[j].cpu().numpy()
            final = np.concatenate((scan_s, predicted_mask_s, mask_s), axis=1)
            final = (final * 255.0).astype(np.uint8)
            cv2.imwrite(scan_path, final)
            slice_id += 1
            # cv2.imshow(scan_id, final)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


@torch.no_grad()
def eval_model(segmentor, discriminator, epoch, exp):
    """ Computing model accuracy """
    correct = 0
    total = 0
    d_loss_list = []
    s_loss_list_source = []
    s_loss_list_target = []
    dice_score_source_list = []
    dice_score_target_list = []
    score_source_list = []
    score_target_list = []
    sdat_source_list = []
    sdat_target_list = []
    hd_source_list = []
    hd_target_list = []
    sdat_scanwise = {"source": [], "target": []}
    segmentor_loss = DiceBCELoss().to(device)
    discriminator_loss = torch.nn.BCELoss().to(device)
    dice_score_func = DiceScore().to(device)

    folder_name = source_domain + "_" + target_domain
    filename_path = os.path.join("data", folder_name, "validation" + ".txt")
    validation_file_df = pd.read_csv(filename_path, header=None)
    validation_file_df["id"] = validation_file_df[0].str[:6]
    scan_id_map = validation_file_df.groupby("id").indices
    validation_file_df["source"] = validation_file_df[0].str.contains(source_domain)
    scan_source_bool = validation_file_df.groupby("id").agg(lambda x: set(list(x))).to_dict()["source"]
    for scan_id in scan_id_map:
        scan_dataset_eval = ScanDataset("validation", scan_id_map[scan_id])
        dataloader_eval = DataLoader(scan_dataset_eval, batch_size=batch_size_train, shuffle=False, num_workers=16)
        progress_bar = tqdm(enumerate(dataloader_eval), total=len(dataloader_eval))
        scan_is_source = list(scan_source_bool[scan_id])[0]
        predicted_mask_list = []
        mask_list = []
        scan_list = []
        for i, batch in progress_bar:
            scans = batch[0].float().to(device)
            masks = batch[1].float().to(device)
            labels = batch[2].float().to(device)

            predicted_mask, x3 = segmentor(scans)
            s_loss = segmentor_loss(predicted_mask, masks)
            predicted_mask[predicted_mask <= 0.5] = 0
            predicted_mask[predicted_mask > 0.5] = 1
            predicted_mask_list.append(predicted_mask)
            mask_list.append(masks)
            scan_list.append(scans)
            dice_score = dice_score_func(predicted_mask, masks)

            if scan_is_source:
                dice_score_source_list.append(dice_score.item())
                s_loss_list_source.append(s_loss.item())
            else:
                dice_score_target_list.append(dice_score.item())
                s_loss_list_target.append(s_loss.item())

            predicted_label = discriminator(x3)
            prediction = torch.zeros_like(labels)
            prediction[torch.where(predicted_label >= 0.5)] = 1
            d_loss = discriminator_loss(predicted_label, labels)
            d_loss_list.append(d_loss.item())

            # Get predictions from the maximum value
            correct += len(torch.where(labels == prediction)[0])
            total += len(labels)

        mask_volume = torch.cat(mask_list, 0).bool().squeeze(1).cpu().numpy()
        predicted_mask_volume = torch.cat(predicted_mask_list, 0).bool().squeeze(1).cpu().numpy()
        score = surf_dst.compute_dice_coefficient(mask_volume, predicted_mask_volume)
        sds = surf_dst.compute_surface_distances(mask_volume, predicted_mask_volume, [1, 1, 1])
        asd = surf_dst.compute_average_surface_distance(sds)
        sdat = surf_dst.compute_surface_dice_at_tolerance(sds, 1)
        hd = surf_dst.compute_robust_hausdorff(sds, 95)
        sdat_scanwise[scan_id] = sdat
        if scan_is_source:
            score_source_list.append(score)
            sdat_source_list.append(sdat)
            hd_source_list.append(hd)
            sdat_scanwise["source"].append({scan_id: sdat})
        else:
            score_target_list.append(score)
            sdat_target_list.append(sdat)
            hd_target_list.append(hd)
            sdat_scanwise["target"].append({scan_id: sdat})

        save_scan(scan_list, mask_list, predicted_mask_list, scan_id, epoch, exp)

    # Total correct predictions and loss
    accuracy = correct / total * 100
    d_loss_mean = np.mean(d_loss_list)
    s_loss_source_mean = np.mean(s_loss_list_source)
    s_loss_target_mean = np.mean(s_loss_list_target)
    dice_score_source_mean = np.mean(dice_score_source_list)
    dice_score_target_mean = np.mean(dice_score_target_list)
    score_source_mean = np.mean(score_source_list)
    sdat_source_mean = np.mean(sdat_source_list)
    hd_source_mean = np.mean(hd_source_list)
    score_target_mean = np.mean(score_target_list)
    sdat_target_mean = np.mean(sdat_target_list)
    hd_target_mean = np.mean(hd_target_list)

    save_sdat(sdat_scanwise, epoch, exp)
    eval_dict = {
        "accuracy": accuracy,
        "d_loss": d_loss_mean,
        "s_loss_source": s_loss_source_mean,
        "s_loss_target": s_loss_target_mean,
        "dice_score_source": dice_score_source_mean,
        "dice_score_target": dice_score_target_mean,
        "score_source": score_source_mean,
        "sdat_source": sdat_source_mean,
        "hd_source": hd_source_mean,
        "score_target": score_target_mean,
        "sdat_target": sdat_target_mean,
        "hd_target": hd_target_mean
    }

    return eval_dict


def train_model():
    exp_id = save_config()
    LR_a = config.getfloat("Classification", "LR_a")
    LR_d = config.getfloat("Classification", "LR_d")
    LR_s = config.getfloat("Classification", "LR_s")
    EPOCHS = config.getint("Classification", "EPOCHS")
    EVAL_FREQ = config.getint("Classification", "EVAL_FREQ")
    SAVE_FREQ = config.getint("Classification", "SAVE_FREQ")
    lamda = config.getfloat("Classification", "lamda")
    model = config.get("Classification", "model")
    segmentor_path = os.path.join("models", "segmentor.pkl")
    discriminator_path = os.path.join("models","discriminator.pkl")
    # initialize model
    # for using with optical flow change modality to "optical_flow"
    segmentor = torch.load(segmentor_path)
    discriminator = torch.load(discriminator_path)
    segmentor_loss = DiceBCELoss().to(device)
    discriminator_loss = torch.nn.BCELoss().to(device)  # cross entropy loss

    # params = list(discriminator.parameters()) + list(segmentor.parameters())
    # optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=LR_d)
    encoder_names = encoder_names = ["init_path","down1","down2","down3"]
    encoder_list = []
    for c in segmentor.named_children():
        if c[0] in encoder_names:
            for p in c[1].parameters():
                encoder_list+=[p]
    optimizer_s = torch.optim.Adam(params=encoder_list, lr=LR_s)  # define optimizer
    # accuracy, d_loss_mean, dice_score_source_mean, dice_score_target_mean, s_loss_source_mean, s_loss_target_mean
    stats = {
        "epoch": [],
        "a_loss": [],
        "s_loss": [],
        "d_loss": [],
        "valid_d_loss": [],
        "accuracy": [],
        "dice_score_source": [],
        "dice_score_target": [],
        "valid_s_loss_source": [],
        "valid_s_loss_target": [],
        "score_source": [],
        "sdat_source": [],
        "hd_source": [],
        "score_target": [],
        "sdat_target": [],
        "hd_target": []
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
            source_indices = torch.where(labels == 1)[0]

            # optimizer_d.zero_grad()  # remove old grads
            optimizer_s.zero_grad()

            # get segmentor loss only for source domains
            predicted_masks, x3 = segmentor(scans)  # get predictions
            # get discriminator loss for source and target
            predicted_labels = discriminator(x3)
            for k, l in enumerate(labels):
                rand = torch.rand(1)
                if rand[0] > 0.5:
                    labels[k] = torch.logical_not(labels[k])
            d_loss = discriminator_loss(predicted_labels, labels)

            # predicted_masks, x3 = segmentor(scans)
            if source_indices.shape[0] > 0:
                predicted_masks_source = predicted_masks[source_indices]
                mask_source = masks[source_indices]
                s_loss = segmentor_loss(predicted_masks_source, mask_source)
                total_loss = s_loss + d_loss
            else:
                total_loss = d_loss

            total_loss.backward()
            optimizer_s.step()
            # update loss lists
            try:
                d_loss_list.append(d_loss.item())
                s_loss_list.append(s_loss.item())
            except:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            try:
                progress_bar.set_description(f"Epoch {0 + epoch} Iter {i + 1}: S loss {s_loss.item():.5f}. ")
                del predicted_labels, predicted_masks, x3, d_loss, s_loss
            except:
                pass

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
            ev_d = eval_model(segmentor, discriminator, epoch, exp_id)
            stats["dice_score_source"].append(ev_d["dice_score_source"] * 100)
            stats["dice_score_target"].append(ev_d["dice_score_target"] * 100)
            stats["accuracy"].append(ev_d["accuracy"])
            stats["valid_d_loss"].append(ev_d["d_loss"])
            stats["valid_s_loss_source"].append(ev_d["s_loss_source"])
            stats["valid_s_loss_target"].append(ev_d["s_loss_target"])
            stats["score_source"].append(ev_d["score_source"])
            stats["sdat_source"].append(ev_d["sdat_source"])
            stats["hd_source"].append(ev_d["hd_source"])
            stats["score_target"].append(ev_d["score_target"])
            stats["sdat_target"].append(ev_d["sdat_target"])
            stats["hd_target"].append(ev_d["hd_target"])
            print(f"Accuracy at epoch {epoch}: {round(ev_d['accuracy'], 2)}%")
            print(f"SDAT Score Source at epoch {epoch}: {round(ev_d['sdat_source'] * 100, 2)}%")
            print(f"SDAT Score Target at epoch {epoch}: {round(ev_d['sdat_target'] * 100, 2)}%")
        if epoch % SAVE_FREQ == 0:
            save_model([segmentor, discriminator], stats, exp_id)


def plot_results(source_domain, target_domain):
    model = config.get("Classification", "model")
    model_name = model + ".pth"
    model_dict = torch.load("models/" + model_name)
    plt.plot(model_dict["stats"]["sdat_source"], label="SDAT Source")
    plt.plot(model_dict["stats"]["sdat_target"], label="SDAT Target")
    # plt.plot(model_dict["stats"]["d_loss"], label="Discriminator Loss")
    plt.suptitle(source_domain + " : " + target_domain + " SDAT Score", fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    folder_name = source_domain + "_" + target_domain
    folder_path = os.path.join("figures", folder_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    loss_path = os.path.join(folder_path, "sdat.jpg")
    plt.legend()
    plt.savefig(loss_path)
    plt.clf()
    plt.cla()
    plt.close()

    # plt.plot(model_dict["stats"]["valid_d_loss"], label="Validation Discrimination Loss")
    # plt.plot(model_dict["stats"]["valid_s_loss_source"], label="Validation Segmentation Loss Source")
    # plt.plot(model_dict["stats"]["valid_s_loss_target"], label="Validation Segmentation Loss Target")
    # plt.suptitle(source_domain + " : " + target_domain + " Validation Loss", fontsize=15)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Loss', fontsize=12)
    # folder_name = source_domain + "_" + target_domain
    # folder_path = os.path.join("figures", folder_name)
    # Path(folder_path).mkdir(parents=True, exist_ok=True)
    # loss_path = os.path.join(folder_path, "validation_loss.jpg")
    # plt.legend()
    # plt.savefig(loss_path)
    # plt.clf()
    # plt.cla()
    # plt.close()
    #
    # plt.plot(model_dict["stats"]["dice_score_source"], label="Source Dice Score")
    # plt.plot(model_dict["stats"]["dice_score_target"], label="Target Dice Score")
    # plt.suptitle(source_domain + " : " + target_domain + " Target Dice Score", fontsize=15)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Dice Score', fontsize=12)
    # folder_name = source_domain + "_" + target_domain
    # folder_path = os.path.join("figures", folder_name)
    # Path(folder_path).mkdir(parents=True, exist_ok=True)
    # loss_path = os.path.join(folder_path, "target_dice_score.jpg")
    # plt.legend()
    # plt.savefig(loss_path)
    # plt.clf()
    # plt.cla()
    # plt.close()
    #
    # plt.plot(model_dict["stats"]["accuracy"], label="Accuracy")
    # plt.suptitle(source_domain + " : " + target_domain + " Validation Accuracy", fontsize=15)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Dice Score', fontsize=12)
    # folder_name = source_domain + "_" + target_domain
    # folder_path = os.path.join("..", "figures", folder_name)
    # Path(folder_path).mkdir(parents=True, exist_ok=True)
    # loss_path = os.path.join(folder_path, "accuracy.jpg")
    # plt.legend()
    # plt.savefig(loss_path)
    # plt.clf()
    # plt.cla()
    # plt.close()

train_model()
# plot_results("ge", "philips")
