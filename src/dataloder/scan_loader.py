import torch
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
import os




def make_filepath_list(source_domain, target_domain, scan_type):
    silver = '/Users/sanketshah/unsup-segmentation/data/Silver-standard'
    filename_path = []
    for filename in os.listdir(silver):
        if ((source_domain in filename) or (target_domain in filename)) and (filename.split('_')[2] == scan_type):
            filename_path.append(os.path.join(silver, filename))
    return filename_path

class ScanDataset(Dataset):
    def __init__(self, source_domain, target_domain, scan_type):
        # load all nii handle in a list
        self.path = make_filepath_list(source_domain, target_domain, scan_type)
        self.images_list = [nib.load(image_path) for image_path in self.path]
        self.domain_list = [image_path.split("_")[1] for image_path in self.path]
        self.source_domain = source_domain
        self.target_domain = target_domain

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        nii_image = self.images_list[idx]
        data = torch.from_numpy(np.asarray(nii_image.dataobj))
        if self.domain_list[idx] == self.source_domain:
            target = 1
        else:
            target = 0
        return data, target

