#!/bin/bash

IMG_SCRIPT="../src/gen_img_list.py"

DATA_DIR="../data/"
G_OPTION="all"
B_OPTION=40
# M_OPTION="svm"
# S_OPTION=0
O_OPTION="../output/"

# python $IMG_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s -1 -o $O_OPTION
python $IMG_SCRIPT -d $DATA_DIR -g $G_OPTION -s -1 -o $O_OPTION
# subject -1 will import all data