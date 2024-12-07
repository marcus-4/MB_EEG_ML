#!/bin/bash

DWNLD_SCRIPT="../src/imageNet_download.py"
IMG_SCRIPT="../src/gen_img_list.py"
CLIP_SCRIPT="../src/blip_clip.py"


DATA_DIR="../data/"
G_OPTION="all"
B_OPTION=40
# M_OPTION="svm"
# S_OPTION=0
O_OPTION="../output/"

python $DWNLD_SCRIPT
# python $IMG_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s -1 -o $O_OPTION
python $IMG_SCRIPT -d $DATA_DIR -g $G_OPTION -s -1 -o $O_OPTION
# subject -1 will import all data

python $CLIP_SCRIPT -d "../data/" -g "all" -m "svm" -b $B_OPTION -s -1 -o "../output/"
