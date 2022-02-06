import torch
import nibabel as nib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
import configparser

os.chdir(os.path.join(".."))
config = configparser.ConfigParser()
config.read(os.path.join("src", "configs", "experiment.config"))
scan_path = config.get("Dataloader", "scan_path")
mask_path = config.get("Dataloader", "mask_path")
source_domain = config.get("Dataloader", "source_domain")
target_domain = config.get("Dataloader", "target_domain")
scan_type = config.get("Dataloader", "scan_type")
batch_size_train = config.getint("Dataloader", "batch_size_train")
batch_size_eval = config.getint("Dataloader", "batch_size_eval")


def get_scan_list(mode, source_domain, target_domain):
    folder_name = source_domain + "_" + target_domain
    filename_path = os.path.join("data", folder_name, mode + ".txt")
    with open(filename_path, "r") as f:
        lines = f.read().split("\n")[:-1]
    return np.array(lines)


def get_transform(pad_size):
    pad = transforms.Pad(pad_size, fill=0, padding_mode="constant")
    # to_tensor = transforms.ToTensor()
    # normalize = transforms.Normalize([0.5], [0.5])
    return transforms.Compose([pad])


def read_nib(nib_path, nib_name):
    nii = nib.load(os.path.join(nib_path, nib_name))
    tensor = torch.from_numpy(np.asarray(nii.dataobj))
    return tensor


class ScanDataset(Dataset):
    def __init__(self, mode, scan_idx=None):
        # load all nii handle in a list
        self.scan_list = get_scan_list(mode, source_domain, target_domain)
        if mode == "validation":
            self.scan_list = self.scan_list[scan_idx]
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.max_pad = 288
        self.mode = mode

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, idx):
        # get name of the scan text file
        scan_text = self.scan_list[idx]
        # get slice id
        slice_id = int(scan_text.split(",")[-1])
        # get domain name of scan
        domain_name = scan_text.split("_")[1]
        # get full scan name
        scan_name = scan_text.split(",")[0]
        # get mask filename
        mask_name = scan_name.split(".nii")[0] + "_ss.nii.gz"
        # read mask and scan
        scan = read_nib(scan_path, scan_name)[slice_id]
        mask = read_nib(mask_path, mask_name)[slice_id]
        # get size of scan
        scan_size = scan.shape[-1]
        # normalize scan
        normalized_scan = scan / scan.max()
        # get transform
        transform = get_transform(int((self.max_pad - scan_size) / 2))
        # apply transform for scan and mask
        scan = transform(normalized_scan).unsqueeze(0)
        mask = transform(mask).unsqueeze(0)
        # get label for scan
        if domain_name == source_domain:
            label = torch.tensor([1])
        elif domain_name == target_domain:
            label = torch.tensor([0])
        return scan, mask, label
