import pdb
import torch
import torch.nn as nn
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torchvision
from PIL import Image
import os.path as osp
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics






def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(" ")
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


def data_load_list(args, p_list):
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    source_set = ObjectImage_list(p_list, train_transform)
    dset_loaders_known = torch.utils.data.DataLoader(
        source_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker,
        drop_last=True,
    )
    return dset_loaders_known


def default_loader(path):
    return Image.open(path).convert("RGB")



IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class ObjectImage_list(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None, loader=default_loader):
        self.imgs = data_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)










def identify_unkown_idx(residual_score, all_output_raw):
    energy = torch.logsumexp(all_output_raw, dim=-1)
    fea = torch.cat((torch.from_numpy(residual_score).unsqueeze(1), energy.unsqueeze(1)), dim=1)
    fea = fea.float().cpu()
    kmeans = KMeans(n_clusters=2, max_iter=500, n_init=20, random_state=42)
    kmeans.fit(fea)
    cluster_0_indices = np.where(kmeans.labels_ == 0)[0]
    cluster_1_indices = np.where(kmeans.labels_ == 1)[0]
    energy_cluster_0 = energy[cluster_0_indices].mean()
    energy_cluster_1 = energy[cluster_1_indices].mean()
    if energy_cluster_0 > energy_cluster_1:
        unknown_idx_all = np.where(kmeans.labels_ == 1)[0]
    else:
        unknown_idx_all = np.where(kmeans.labels_ == 0)[0]
    return unknown_idx_all
