import os
import random

import torch
import torch.utils.data as data

from preprocess.image_preprocess import ImagePreprocess

class Dataset(data.Dataset):
    def __init__(self, data_dir, resize_to):
        # important variables
        self.file_list = []
        self.labels = []
        self.preprocessor = ImagePreprocess()
        self.resize_to = resize_to
        # forming current data dir
        # --------
        cur_dir = os.getcwd()
        cur_dir = cur_dir.replace("\\", "/") # "path\\path" -> "path/path"
        cur_dir += "/data/"  # workdir/model/data
        # --------
        self.data_dir = cur_dir + data_dir # train, val, test
        # --------
        fake_files = os.listdir(self.data_dir + "/Fake/") # data_dir/Fake/
        real_files = os.listdir(self.data_dir + "/Real/") # data_dir/Real/ ------------------ correct the paths
        # --------
        file_names = []
        file_names.extend(fake_files)
        file_names.extend(real_files)

        random.shuffle(file_names) # model will be overstudied if fake labels first then real ones
        random.shuffle(file_names) # model will be overstudied if fake labels first then real ones

        for i in range(len(file_names)):
            if "real" in file_names[i]:
                self.labels.append(1)
                self.file_list.append(os.path.join(self.data_dir + "/Real/", file_names[i]))
            elif "fake" in file_names[i]:
                self.labels.append(0)
                self.file_list.append(os.path.join(self.data_dir + "/Fake/", file_names[i]))
        # --------
        print(self.file_list[:5])
        print(self.labels[:5])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image = self.preprocessor.read_image(self.file_names[index], with_channels=True)
        image = self.preprocessor.im_preprocess(image, self.resize_to)
        return image