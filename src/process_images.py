#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------------------
# Created By Marcus Becker

# This program rescales all images used and places only those being used into a new folder
# --------------------------------------------------------------------------------------------



import cv2
import os

# image = np.asarray(bytearray(resp.read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# image = cv2.resize(image,(224,224))

new_path =  "../data/imageNet_images/"
old_path = "/Users/marcus/Desktop/575/project/EEG-ImageNet-Dataset/data/imageNet_images_old/"
# old_path = "../data/raw_imageNet_images/"

output_size = 224


def convert_image(path):
    old_image_file = os.path.join(old_path, path.split('_')[0], path)
    new_image_file = os.path.join(new_path, path.split('_')[0], path)

    new_synset_path = os.path.join(new_path, path.split('_')[0])
    if not os.path.exists(new_synset_path):
        os.makedirs(new_synset_path)

    im = cv2.imread(old_image_file, cv2.IMREAD_COLOR)
    if im is None:
        print(f"Error: Could not read image {old_image_file}")
        return False

    try:
        im_resize = cv2.resize(im, (output_size, output_size))
        cv2.imwrite(new_image_file, im_resize)
        print(f"Converted and saved: {new_image_file}")
        return True
    except Exception as e:
        print(f"Error processing {old_image_file}: {e}")

def image_exists(path):
    old_image_file = os.path.join(old_path, path.split('_')[0], path)
    new_image_file = os.path.join(new_path, path.split('_')[0], path)
    with open(os.path.join("../output", f"ommitted.txt"), "r") as f:
                file_contents = f.read()
    if path not in file_contents:
         return False
    if not os.path.exists(new_image_file):
        if not os.path.exists(old_image_file): 
            if path not in file_contents:
                with open(os.path.join("../output", f"ommitted.txt"), "a") as f:
                    f.write(path+"\n")
            return False
        else:
            convert_image(path)
    return True


if __name__ == '__main__':
    convert_image("n07758680_3041.JPEG")
        # convert_image(image_name)

