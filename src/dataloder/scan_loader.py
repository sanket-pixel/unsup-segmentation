import torch
import nibabel as nib
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os


def make_filepath_list(mode, source_domain, target_domain, scan_path):
    folder_name = source_domain+"_"+target_domain
    filename_path = os.path.join("..","data", folder_name,mode+".txt")
    with open(filename_path, "r") as f:
        lines = f.read().split("\n")[:-1]
    path = [os.path.join(scan_path, file) for file in lines]
    return path

class ScanDataset(Dataset):
    def __init__(self, source_domain, target_domain,scan_path,scan_type, mode):
        # load all nii handle in a list
        self.path = make_filepath_list(mode,source_domain,target_domain, scan_path)
        self.images_list = [nib.load(image_path) for image_path in self.path]
        self.domain_list = [image_path.split("_")[1] for image_path in self.path]
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.transform = transforms.Resize(512)
        self.mode = mode

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        nii_image = self.images_list[idx]
        data = torch.from_numpy(np.asarray(nii_image.dataobj))
        data = self.transform(data)
        random_slice = torch.randint(high=data.shape[0], size=(1,))
        data = data[random_slice]
        if self.domain_list[idx] == self.source_domain:
            target = 1
        else:
            target = 0
        return data, target

