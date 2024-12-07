#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------
# Created By Marcus Becker

# This program downloads the relevant synset categories from the imageNet-21k Winter21 dataset
# --------------------------------------------------------------------------------------------



from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import urllib
import matplotlib.pyplot as plt
import random
import os
import socket
import tarfile
from urllib.request import urlopen

def download_tar(url, download_path, extract_to=None):

    try:
        # Download the .tar file
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Save the .tar file
        with open(download_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded and saved to {download_path}")

        # Extract the .tar file if requested
        if extract_to:
            print(f"Extracting {download_path} to {extract_to}...")
            with tarfile.open(download_path, "r") as tar:
                tar.extractall(path=extract_to)
            print(f"Extracted to {extract_to}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except tarfile.TarError as e:
        print(f"Error extracting .tar file: {e}")

def main():       
    startp = 0
    path=os.path.join("../data/wnids1000.txt")
    # read in the wordnet id list
    infile = open(path,"r")
    lines = infile.readlines()
    wnids = []
    for line in lines:
        wnids.append(line.strip('\n').split(' ')[0])
    wnids = wnids[startp:]
    # download images


    for i in range(len(wnids)):
        print("wnids %s"%(wnids[i]))
        url = "https://image-net.org/data/winter21_whole/" + wnids[i] + ".tar"
        download_path = os.path.join("../data/raw_imageNet_images/" + wnids[i] + ".tar")
        extract_to = os.path.join("../data/raw_imageNet_images/" + wnids[i] + "/")


        # url = "https://image-net.org/data/winter21_whole/n03709823.tar"
        # download_path = "/Users/marcus/Desktop/575/project/EEG-ImageNet-Dataset/data/imageNet_images/n03709823.tar"
        # extract_to=None
        # Ensure the extract_to directory exists
        if extract_to and not os.path.exists(extract_to):
            os.makedirs(extract_to)
            download_tar(url, download_path, extract_to)
        

if __name__ == '__main__':
    main()
