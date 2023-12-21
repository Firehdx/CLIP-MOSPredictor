import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

#要修改以获得不同指标
# for 3K
def get_label_json(imgs, path):
    with open(path) as f:
        dic = json.load(f)
    labels = []
    for img in imgs:
        label = dic[img][0]
        labels.append([float(label)])
    return labels

# for 2023
def get_multi_labels_json(imgs, paths):
    multi_labels = np.array(get_label_json(imgs, paths[0]))
    for i in range(1, len(paths)):
        label = np.array(get_label_json(imgs, paths[i]))
        multi_labels = np.concatenate((multi_labels, label), axis=1)
    return multi_labels
        


class CLIPDataset(Dataset):
    def __init__(self, CLIPfeatures, MOSlabels):
        self.features = CLIPfeatures
        self.labels = MOSlabels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]