import os
import random

import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, data_dir):
        # to add all file paths
        self.labels = []
        # forming current data dir
        # --------
        cur_dir = os.getcwd()
        cur_dir = cur_dir.replace("\\", "/") # "path\\path" -> "path/path"
        cur_dir = cur_dir + "/data/" # workdir/model/data
        # --------
        self.data_dir = cur_dir + data_dir # train, val, test
        # --------
        fake_labels = os.listdir(self.data_dir + "/Fake/") # data_dir/Fake/
        real_labels = os.listdir(self.data_dir + "/Real/") # data_dir/Real/
        # --------
        self.labels.extend(fake_labels)
        self.labels.extend(real_labels)
        random.shuffle(self.labels) # model will be overstudied if fake labels first then real ones
        # --------

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        pass