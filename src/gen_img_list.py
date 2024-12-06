# --------------------------------------------------------------------------------------------
# Created By Marcus Becker
# Derived from EEG-ImageNet-Dataset
# https://github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset/tree/main
# --------------------------------------------------------------------------------------------

import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from de_feat_cal import de_feat_cal
from dataset import EEGImageNetDataset
# from model.mlp_sd import MLPMapper
from utilities import *
from process_images import convert_image, image_exists

# default="../data/"
old_path = "/Users/marcus/Desktop/575/project/EEG-ImageNet-Dataset/data/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", default=old_path, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", default="all", help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-m", "--model", required=False, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=-1, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", default="../output/", help="directory to save results")
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset(args)
    #load all, iterate through all, remove missing images, save, reload all 
    for image_name in tqdm(dataset.images):
            image_path = os.path.join("../data/imageNet_images", image_name.split("_")[0], image_name)
            image_exists(image_name)
            # print(image_name)
            # print(image_path)


    #rewrite following to account for potentially missing images

    with open(os.path.join(args.output_dir, f"s{args.subject}.txt"), "w") as f:
        dataset.use_image_label = True
        for data in dataset:
            # f.write(f"{data[1]}\n")
            image_name=data[1]
            # image_path = os.path.join("../data/imageNet_images", image_name.split("_")[0], image_name)
            if image_name == None or (not image_exists(image_name)):
                 dataset.remove_invalid(data)
            else:
                 f.write(f"{data[1]}\n")
        dataset.save_cleaned_dataset()

    # with open(os.path.join(args.output_dir, f"s{args.subject}_label.txt"), "w") as f:
    #     dataset.use_image_label = False
    #     for idx, data in enumerate(dataset):
    #         if idx % 50 == 0:
    #             label_wnid = dataset.labels[data[1]]
    #             f.write(f"{idx + 1}-{idx + 50}: {wnid2category(label_wnid, 'ch')}\n")