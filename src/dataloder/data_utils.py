import os
import random
from matplotlib import pyplot as plt
import torch
from pathlib import Path
import configparser
from random import shuffle
import nibabel as nib
import numpy as np

os.chdir(os.path.join("..", ".."))

config = configparser.ConfigParser()
config.read(os.path.join("src", "configs", "scan_seg.config"))
scan_path = config.get("Dataloader", "scan_path")
mask_path = config.get("Dataloader", "mask_path")
source_domain = config.get("Dataloader", "source_domain")
target_domain = config.get("Dataloader", "target_domain")
scan_type = config.get("Dataloader", "scan_type")


def make_scan_slide_list(scan_list):
    scan_slice_id_list = []
    for scan in scan_list:
        scan_nii = nib.load(os.path.join(scan_path, scan))
        scan_tensor = torch.from_numpy(np.asarray(scan_nii.dataobj))
        total_slices = scan_tensor.shape[0]
        for i in range(total_slices):
            scan_slice_id = scan + "," + str(i)
            scan_slice_id_list.append(scan_slice_id)
    return scan_slice_id_list


def make_text_file(scan_list, file_name):
    folder_name = source_domain + "_" + target_domain
    Path(os.path.join("data", folder_name)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join("data", folder_name, file_name), "w") as text_file:
        for file in scan_list:
            text_file.write(file)
            text_file.write("\n")
    text_file.close()


def save_scans_text(scan_list, file_name):
    scan_id_list = make_scan_slide_list(scan_list)
    make_text_file(scan_id_list, file_name)


# get total scans with source and target domains
total_scans = []
for filename in os.listdir(scan_path):
    if (source_domain in filename or target_domain in filename) and (filename.split('_')[2] == scan_type):
        total_scans.append(filename)
# make train val test split
# for each scan, get number of slices
shuffle(total_scans)
train_idx = int(len(total_scans) * 0.7)
val_idx = int(len(total_scans) * 0.9)
train_scans = total_scans[:train_idx]
val_scans = total_scans[train_idx:val_idx]
test_scans = total_scans[val_idx:]
# add a row for each scan's each slice
save_scans_text(train_scans, "training.txt")
save_scans_text(val_scans, "validation.txt")
save_scans_text(test_scans, "testing.txt")
