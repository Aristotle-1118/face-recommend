import sys, torch, random
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append("/home/rilea/exp0519/tuxiangchuli/task4")

from torch.utils.data import Dataset
import os

import pandas as pd
import numpy as np
from PIL import Image
from torch import tensor
torch.set_default_tensor_type(torch.DoubleTensor)
print("start to create dataset")

class faceDataset(Dataset):
    def __init__(self, img_dir, rating_path, transform=None, train_or_test=None ):
        self.img_dir = img_dir
        self.rating_path = rating_path
        self.rating_path = np.array(pd.read_csv(self.rating_path, usecols=[1,2]))
        self.transform = transform
        self.train_or_test = train_or_test

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        # img_name = self.img_path[idx]

        img_name = "SCUT-FBP-"+str(idx+1)+".jpg" if self.train_or_test == "train" else "SCUT-FBP-"+str(idx+1+400)+".jpg"
        img_item_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_item_path)
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(np.array(img)).double()
        rating_mean = self.rating_path[:,0][idx]*20
        rating_standard = self.rating_path[:,1][idx]
        item = {"img_name": img_name, "img": img, "rating_mean": rating_mean, "rating_standard": rating_standard}

        # return item
        return img_name, img, rating_mean, rating_standard


