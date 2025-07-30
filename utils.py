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
from data_list import ImageList, ImageList_idx, ImageList_idx_aug, ImageList_idx_aug_fix
from torchvision import transforms
from torch.utils.data import DataLoader



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




def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}

        src_classes = list(range(len(args.src_classes)))

        for i in range(len(src_classes)):
            label_map_s[src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items(): 
        s += "{}:{}\n".format(arg, content)
    return s



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



