import os
import json
import numpy as np
import matplotlib.pyplot as plt
import clip
import pathlib
import sklearn.preprocessing
import torch
from torch.utils.data import DataLoader
import dataset
from net import get_model
from train import train
import clipscore

if __name__ == '__main__':
    workers = 6
    num_epochs = 30
    lr = 0.01
    image_dir = "AIGCIQA2023/Image/allimg/"
    candidates_json = "AIGCIQA2023/DATA/prompts.json"
    qua_json = "AIGCIQA2023/DATA/qua.json"
    aut_json = "AIGCIQA2023/DATA/aut.json"
    cor_json = "AIGCIQA2023/DATA/cor.json"
    json_paths = [qua_json,aut_json,cor_json]


    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    image_ids = [pathlib.Path(path).name for path in image_paths]

    with open(candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = clipscore.extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=workers)
    prompt_feats = clipscore.extract_all_captions(
        candidates, model, device, batch_size=256, num_workers=workers)
    
    CLIPfeats = np.concatenate((image_feats, prompt_feats), axis=1)
    MOSlabels = np.array(dataset.get_multi_labels_json(image_ids, json_paths))
    MOSlabels = MOSlabels[:,0].reshape(-1,1)
    CLIPfeats = sklearn.preprocessing.normalize(CLIPfeats, axis=1)
    MOSlabels = sklearn.preprocessing.normalize(MOSlabels, axis=1)

    train_nums = 2000

    train_feats = CLIPfeats[:train_nums]
    train_labels = MOSlabels[:train_nums]
    train_set = dataset.CLIPDataset(train_feats, train_labels)

    # validate_feats = CLIPfeats[train_nums:train_nums+validate_nums]
    # validate_labels = MOSlabels[train_nums:train_nums+validate_nums]
    # validate_set = dataset.CLIPDataset(validate_feats, validate_labels)

    test_feats = CLIPfeats[train_nums:]
    test_labels = MOSlabels[train_nums:]
    test_set = dataset.CLIPDataset(test_feats, test_labels)

    train_loader = DataLoader(train_set, batch_size=32, num_workers=workers)
    # validate_loader = DataLoader(validate_set, batch_size=16, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=16, num_workers=workers)

    net = get_model(CLIPfeats.shape[1], MOSlabels.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_loader, test_loader, criterion=criterion, optimizer=optimizer, 
            num_epochs=num_epochs, model_name="2023_MLP_512_128_qua.pth")