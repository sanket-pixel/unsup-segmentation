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


def make_filepath_list(mode, source_domain, target_domain, scan_path, mask_path):
    folder_name = source_domain + "_" + target_domain
    filename_path = os.path.join("..", "data", folder_name, mode + ".txt")
    with open(filename_path, "r") as f:
        lines = f.read().split("\n")[:-1]
    scan_path = [os.path.join(scan_path, file) for file in lines]
    mask_path = [os.path.join(mask_path, file.split(".nii")[0] + "_ss.nii.gz") for file in lines]
    return scan_path, mask_path


def get_transform(pad_size):
    pad = transforms.Pad(pad_size, fill=0, padding_mode="constant")
    # to_tensor = transforms.ToTensor()
    # normalize = transforms.Normalize([0.5], [0.5])
    return transforms.Compose([pad])


class ScanDataset(Dataset):
    def __init__(self, source_domain, target_domain, scan_path, mask_path,
                 scan_type, mode):
        # load all nii handle in a list
        self.scan_path, self.mask_path = make_filepath_list(mode, source_domain,
                                                            target_domain, scan_path, mask_path)
        self.scan_list = [nib.load(image_path) for image_path in self.scan_path]
        self.mask_list = [nib.load(image_path) for image_path in self.mask_path]
        self.domain_list = [image_path.split("_")[1] for image_path in self.scan_path]

        self.source_domain = source_domain
        self.target_domain = target_domain
        self.num_slices = 2
        self.max_pad = 288
        self.mode = mode
        self.scan_list_t, self.mask_list_t, self.label_list = self.transform_concat(self.scan_list, self.mask_list)

    def __len__(self):
        return len(self.scan_path)

    def transform_concat(self, scan_list, mask_list):
        scan_transformed_list = []
        mask_transformed_list = []
        label_list = []
        for i, scan_nii in enumerate(scan_list):
            scan_tensor = torch.from_numpy(np.asarray(scan_nii.dataobj))
            mask_tensor = torch.from_numpy(np.asarray(mask_list[i].dataobj))
            scan_size = torch.Tensor(list(scan_tensor.shape))
            if scan_size[0] == scan_size[1]:
                scan_tensor = scan_tensor.permute(2, 0, 1)
                mask_tensor = mask_tensor.permute(2, 0, 1)
            elif scan_size[0] == scan_size[2]:
                scan_tensor = scan_tensor.permute(1, 0, 2)
                mask_tensor = mask_tensor.permute(1, 0, 2)
            scan_size = scan_tensor.shape[-1]
            max_across_slice_scan = scan_tensor.amax(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
            normalized_scan = scan_tensor / max_across_slice_scan
            transform = get_transform(int((self.max_pad - scan_size) / 2))
            scan = transform(normalized_scan)
            mask = transform(mask_tensor)
            total_slices = scan_tensor.shape[0]
            if self.domain_list[i] == self.source_domain:
                label_list.append(torch.ones(total_slices, 1))
            elif self.domain_list[i] == self.target_domain:
                label_list.append(torch.zeros(total_slices, 1))
            scan_transformed_list.append(scan)
            mask_transformed_list.append(mask)

        return torch.cat(scan_transformed_list, 0), torch.cat(mask_transformed_list, 0), torch.cat(label_list, 0)

    def __getitem__(self, idx):
        scan = self.scan_list_t[idx]
        mask = self.mask_list_t[idx]
        label = self.label_list[idx]
        return scan, mask, label

# source_domain = "siemens"
# target_domain = "philips"
# scan_path = "../../data/Original/Original"
# mask_path = "../../data/Silver-standard-machine-learning/Silver-standard"
# scan_type = "3"
# scan_dataset_train = ScanDataset(source_domain, target_domain,
#                                  scan_path,mask_path, scan_type, "training")
# dataloader_train = DataLoader(scan_dataset_train, batch_size=8, shuffle=True)
#
# for batch in dataloader_train:
#     print(batch[0].shape)
#     print(batch[1].shape)
#     print(batch[2].shape)
#     break
