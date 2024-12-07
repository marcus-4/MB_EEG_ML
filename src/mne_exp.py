# --------------------------------------------------------------------------------------------
# Created By Marcus Becker
# Derived from EEG-ImageNet-Dataset
# https://github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset/tree/main
# --------------------------------------------------------------------------------------------

import numpy as np
import mne
from utilities import *
from dataset import EEGImageNetDataset
import argparse
import sys
import json
from de_feat_cal import de_feat_cal
import matplotlib
# matplotlib.use('macosx')
matplotlib.use('Agg')
import matplotlib.pyplot as plt



channels = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
            "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
            "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2",
            "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]


def get_edge_weight():
    montage = mne.channels.read_dig_fif('../data/mode/' + 'montage.fif')
    montage.ch_names = json.load(open('../data/mode/' + "montage_ch_names.json"))
    edge_pos = montage.get_positions()['ch_pos']
    edge_weight = np.zeros([len(channels), len(channels)])
    edge_pos_value = [edge_pos[key] for key in channels]
    delta = 4710000000
    edge_index = [[], []]
    for i in range(len(channels)):
        for j in range(len(channels)):
            edge_index[0].append(i)
            edge_index[1].append(j)
            if i == j:
                edge_weight[i][j] = 1
            else:
                edge_weight[i][j] = np.sum([(edge_pos_value[i][k] - edge_pos_value[j][k]) ** 2 for k in range(3)])
                edge_weight[i][j] = min(1, delta / edge_weight[i][j])
    global_connections = [['FP1', 'FP2'], ['AF3', 'AF4'], ['F5', 'F6'], ['FC5', 'FC6'], ['C5', 'C6'],
                          ['CP5', 'CP6'], ['P5', 'P6'], ['PO5', 'PO6'], ['O1', 'O2']]
    for item in global_connections:
        i, j = item
        if i in channels and j in channels:
            i = channels.index(item[0])
            j = channels.index(item[1])
            edge_weight[i][j] -= 1
            edge_weight[j][i] -= 1
    return edge_index, edge_weight

if __name__ == '__main__':
    sys.argv = ["blip_clip.py", "-d", "../data/", "-g", "fine4", "-m", "svm","-s", "1", "-o", "../output/"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument("-g", "--granularity", required=True, help="choose from coarse, fine0-fine4 and all")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset(args)
    eeg_data  = np.stack([i[0].numpy() for i in dataset], axis=0)
    # eeg_data = np.load(os.path.join('../data/de_feat/', f"{args.subject}_{args.granularity}_de.npy"))
    # eeg_data = np.load(os.path.join('../data/de_feat/', f"1_coarse_de.npy"))

    # extract frequency domain features
    mne.viz.set_browser_backend('matplotlib')
    
    
    # Simulated EEG data: shape (n_epochs, n_channels, n_times)
    n_epochs = 10  # Number of epochs
    n_times = 1000  # Number of time points per epoch (e.g., 1 second at 1000 Hz sampling rate)
    sfreq = 1000  # Sampling frequency in Hz

    channel_names = [f'EEG{i}' for i in range(1, 63)]
    info = mne.create_info(ch_names=channels, sfreq=1000, ch_types='eeg')
    _epochs = mne.EpochsArray(data=eeg_data, info=info)
    # _epochs.plot()
    montage = mne.channels.read_dig_fif('../data/mode/' + 'montage.fif')
    montage.ch_names = json.load(open('../data/mode/' + "montage_ch_names.json"))
    edge_pos = montage.get_positions()['ch_pos']
    # montage = mne.channels.make_standard_montage('standard_1020')
    _epochs.set_montage(montage)#.rename_channels()
    # Plot the EEG data
    _epochs.plot(scalings='auto', n_channels=62, title="EEG Data", show=True)
    plt.savefig('eeg_plot1.png')
    # Plot the electrode locations on the brain
    montage.plot(kind='3d', show_names=True, sphere=0.1)
    plt.savefig('eeg_plot2.png')
    # Optionally, plot topographical layout of electrodes
    montage.plot(kind='topomap', show_names=True)
    plt.savefig('eeg_plot3.png')




    # de_feat = de_feat_cal(eeg_data, args)
    # dataset.add_frequency_feat(de_feat)

   