import os
import random
from matplotlib import pyplot as plt
import torch
from pathlib import Path


# unet = torch.load("/home/sanket/Desktop/Projects/unsup-segmentation/models/Unet_without_adverserial.pth")
# training_loss  = unet['stats']['train_loss'][:80]
# plt.plot(training_loss)
# plt.title("Average Training Loss")
# plt.savefig("../figures/training_loss.png")

def make_train_val_text(source_domain,target_domain,scan_type):
    silver = '../data/Original/Original'
    total_scans = []
    for filename in os.listdir(silver):
        if (source_domain in filename or target_domain in filename) and  (filename.split('_')[2] == scan_type):
            total_scans.append(filename)

    train_scan_files = random.sample(total_scans,int(len(total_scans)*0.7))
    eval_scan_files = list(set(total_scans).difference(set(train_scan_files)))
    folder_name = source_domain+"_"+target_domain
    Path(os.path.join("..","data",folder_name)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join("..","data",folder_name,"training.txt"), "w") as text_file:
        for file in train_scan_files:
            text_file.write(file)
            text_file.write("\n")
    text_file.close()
    with open(os.path.join("..","data",folder_name,"validation.txt"), "w") as text_file:
        for file in eval_scan_files:
            text_file.write(file)
            text_file.write("\n")
    text_file.close()


source_domain = 'ge'
target_domain = 'phil'
scan_type = '3'
make_train_val_text(source_domain,target_domain,scan_type)