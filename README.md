# Entropy guided Unspervised domain adaptation

## Abstract
Image segmentation is an important task for many medical analytical applications. With the advancement in data and computational resources, deep learning methods have been very effective to solve the segmentation task. However, a segmentation model trained on a particular domain's domain (source) performs poorly on when used on novel (target) domains. To that end, in this lab we implement a paper on unsupervised domain adaptation that introduces an adversarial learning based domain invariant segmentation model. The model trains a segmenter to segment the scans from source domain along with a feature discriminator that discriminates features from source and target domains. Finally their model also consists of entropy discriminator that discriminates the entropy distributions from source and target domains. For adversarial training, the discriminator labels were flipped which enforces the alignment between the features and entropy of source and target. DICE Score and Surface Dice score were used to evaluate the segmentation performance. However, in this lab, we made a few crucial changes to the original model design. Unlike the original paper, we used a U-Net backbone for segmentation. Moreover, the feature discriminator is trained on both source and target domains, unlike the original paper.  

## Guide to run Desktop version

  1. Change the config file ( **src/configs/experiment.config**  ) to tune parameters to change training conditions. The names of the parameters in the config file are self-explainatory.
  2. Run **train.py**
  3. The results will be stored in **results/experiment_n** where after every run a new experiment folder will be added in results containing the model, predicted scans and metrics.

## Guide to run on Google Colab

### View the code at:
 https://colab.research.google.com/drive/1B7T-Fncv7GOjojs39WIO8xT_0_Ddu3Tu?usp=sharing

### Pre-requisites to run
 1. To run the notebooks as is you will need a pro account. Otherwise you will have to make changes to the Dataset based on available memory.
 2. After downloading the CC359 dataset you can upload the scans to your google drive with the folder structure mentioned below:

  - The folder structure used for running this notebook is: 
    - base_path = '/content/gdrive/MyDrive/VMIA_Lab_Data/data/'
    - original_path = base_path + 'Original/Original/'
    - mask_path = base_path + 'Silver-standard-machine-learning/Silver-standard/'

3. Adapt your folder structure accordingly. Alternatively make changes to the paths in the Dataset class of the code.
4. All other instructions and pointers are also provided in the code files.
