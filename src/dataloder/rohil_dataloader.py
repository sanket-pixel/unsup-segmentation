import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import string
from sklearn.preprocessing import MinMaxScaler
import torch
import nibabel as nib
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def process_image(img, max_pad = 288):
     
    '''
    MIN-MAX SCALE THE IMAGE
    '''
    assert (len(img.shape) == 3)
    assert img.min() == 0
    img = img/img.max()
    
    '''
    FIND NON UNIQUE AXIS AND BRING TO 0-TH POS
    '''
    vals, counts = np.unique(img.shape, return_counts = True)
    
    no_of_scans = vals[counts == 1]
    image_size = vals[counts == 2]
    
    assert (len(no_of_scans) == 1) and (len(image_size) == 1)
    img = np.swapaxes( img, 0, np.where(img.shape == no_of_scans)[0][0] )
    
    '''
    PAD IMAGE TO MAX SIZE
    '''
    pad_size = int((max_pad - image_size[0])/2)   
    
    
    return np.pad(img, pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode =  'constant', constant_values = 0)



def make_filepath_list(domain, scan_type):
    base_path = 'C:/Users/rohil/Downloads/Uni Bonn/WiSe 2021-22/Lab Medical Imaging/'
    original_path = 'Data/Original/Original/'
    mask_path = 'Data/Silver-standard-machine-learning/Silver-standard/'

    original_list = []
    mask_list =[]

    for filename in os.listdir(base_path + original_path):
        
        if (domain in filename) and (filename.split('_')[2] == scan_type) and (filename.endswith('.nii.gz')):
            
            original_list.append(base_path + original_path + filename)
            mask_list.append(base_path + mask_path + filename[:-7] + '_ss.nii.gz')

    return original_list, mask_list


class ScanDataset(Dataset):
    def __init__(self, domain, scan_type, mode = 'complete', train_split = None, val_split = None, test_split = None):
        
        self.original_img_path, self.mask_img_path = make_filepath_list(domain, scan_type)
        
        self.original_images = [ process_image(nib.load(image_path).get_fdata()) for image_path in self.original_img_path]
        self.mask_images = [process_image(nib.load(image_path).get_fdata()) for image_path in self.mask_img_path]
        
        print("Total number of files available for",domain,",",scan_type,":", len(self.original_images))
        
        if mode == 'train':   
            
            start_idx = 0
            stop_idx = int(train_split * len(self.original_images))
        
            self.original_images = self.original_images[start_idx:stop_idx]
            self.mask_images = self.mask_images[start_idx:stop_idx]
            
            print("Number of files selected for train:", stop_idx )
            
        elif mode == 'val':
            
            start_idx = int(train_split * len(self.original_images))
            stop_idx = int((train_split + val_split)* len(self.original_images))
            
            self.original_images = self.original_images[start_idx:stop_idx]
            self.mask_images = self.mask_images[start_idx:stop_idx]
            
            print("Number of files selected for val:", stop_idx - start_idx )
        
        elif mode == 'test':
            
            start_idx = int((train_split + val_split)* len(self.original_images))
            print("Number of files selected for test:", len(self.original_images) - start_idx )
            
            self.original_images = self.original_images[start_idx:]
            self.mask_images = self.mask_images[start_idx:]
                        
        
        self.original_images = np.concatenate( self.original_images , axis = 0)
        self.mask_images = np.concatenate( self.mask_images , axis = 0)

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):

        original_batch = torch.from_numpy(self.original_images[idx]).float()
        mask_batch = torch.from_numpy(self.mask_images[idx]).float()

        return original_batch, mask_batch



##### EXAMPLE #####

# TRAIN-VAL-TEST SPLIT
domain = 'ge' 
ptype = '3'

train_data = ScanDataset(domain, ptype ,mode='train', train_split = 0.7, val_split = 0.2, test_split = 0.1)
val_data = ScanDataset(domain, ptype, mode='val', train_split = 0.7, val_split = 0.2, test_split = 0.1)
test_data = ScanDataset(domain, ptype, mode='test', train_split = 0.7, val_split = 0.2, test_split = 0.1)

# LOAD COMPLETE DATASET
domain_l = ['siemens','ge','philips'] 
ptype_l = ['15','3']

for domain in domain_l:
    for ptype in ptype_l:
        if not (domain == 'ge' and ptype == '3'):
            data = ScanDataset(domain, ptype ,mode='complete')
            print(domain,ptype,len(data))
            np.save(domain+ptype+'_original_images', data.original_images.astype('float32'))
            np.save(domain+ptype+'_mask_images', data.mask_images.astype('float32'))