from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import numpy as np


class VecDataset(Dataset):
    def __init__(self, data_df, labels_df, info_df , data_path = "./dataset/mimic_npy/mimic-cxr/files"):
        
        self.data_df = pd.read_csv(data_df)
        self.labels_df = pd.read_csv(labels_df)
        self.info_df = pd.read_csv(info_df)

        self.data = self.data_df.to_numpy()
        self.labels = self.labels_df.to_numpy()
        self.info = self.info_df.to_numpy()
        self.data_path = data_path
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # read .npy file with path in  data_df
        item_path = self.data[idx][0]

        full_item_path = self.data_path + item_path + ".npy"

        item = np.load(full_item_path)
        #convert to tensor 
        item = torch.from_numpy(item)
        # read labels from labels_df
        label = list(self.labels[idx])
        label = torch.tensor(label)

        return {'data':item, 'labels': label}


    






