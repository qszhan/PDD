import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader

import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from numpy import linalg as LA
from collections import Counter
from hsic import hsic_normalized
from loss import CrossEntropyLabelSmooth
from utils import make_dataset, data_load_list, identify_unkown_idx, data_load, print_args
import random
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sam import SAM
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance


 

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer



def whole_net_pass(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas_F = netF(inputs)
            feas = netB(feas_F)
            outputs = netC(feas)
            if start_test:
                all_fea_F = feas_F.float().cpu()
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea_F = torch.cat((all_fea_F, feas_F.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    kn_un_labels = (all_label == args.class_num).float()
    return all_fea_F, all_fea, all_output, all_label, kn_un_labels


def obtain_cre_domain(loader, netF, netB, netC, args, iter_num_update_f, easy_idx, easy_path):
    """ the full (path, label) list """
    img = make_dataset('', args.t_dset_path)
    all_fea_F, all_fea, all_output_raw, all_label, kn_un_labels = whole_net_pass(loader, netF, netB, netC, args)
    energy = torch.logsumexp(all_output_raw, dim=-1)
    all_output = nn.Softmax(dim=1)(all_output_raw)
    _, predict = torch.max(all_output, 1)
    len_confi = int(energy.shape[0] * args.sigma)
    len_confi = min(len_confi, energy.shape[0])
    idx_confi = energy.topk(len_confi, largest=True)[-1]
    idx_confi_list_ener = idx_confi.cpu().numpy().tolist()
    label_energy = np.zeros(all_label.shape[0], dtype="int64")
    label_energy[idx_confi_list_ener] = 1

    """ extract the prototypes """
    labelset = list(range(args.class_num))
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label_with_dis_energy = [(pred_label[i], dd[i, pred_label[i]], label_energy[i]) for i in
                                  range(len(pred_label))]
    closest_samples = {label: [] for label in labelset}
    top_num = {label: [] for label in labelset}
    top_num_samples = {label: [] for label in labelset}
    num_selected = int(args.confi_nums * (1 + args.con_num_growth * iter_num_update_f))
    for i, (label, distance, energy) in enumerate(pred_label_with_dis_energy):
        closest_samples[label].append((i, distance, energy))
    for label in labelset:
        top_num[label] = [item for item in closest_samples[label] if item[2] == 1][:num_selected]
        easy_idx = list(set(easy_idx + [item[0] for item in top_num[label]]))
        top_num_samples[label] = [item for item in easy_idx]
        for idx in easy_idx:
            easy_path.append((img[idx][0], pred_label[idx]))

    # update class prototypes according to top_num_samples
    selet_fea = all_fea[easy_idx]  # (209, 257)
    selet_aff = all_output[easy_idx].float().cpu().numpy()
    polished_initc = selet_aff.transpose().dot(selet_fea)
    polished_initc = polished_initc / (1e-8 + selet_aff.sum(axis=0)[:, None])
    return pred_label.astype(
        'int'), all_fea_F, all_fea, all_output_raw, polished_initc, kn_un_labels, easy_idx, easy_path


def obtain_nearest_prototype(data_q, polished_initc):
    data_q_ = data_q
    data_q_ = (data_q_.t() / torch.norm(data_q_, p=2, dim=1)).t()
    data_q_ = data_q_.detach().float().cpu().numpy()
    dd = cdist(data_q_, polished_initc, args.distance)
    pred_pyoto = dd.argmin(axis=1)
    initc_nei = polished_initc[pred_pyoto]
    return initc_nei, pred_pyoto


def cal_acc_oda(loader, netF, netB, netC, easy_idx):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output_raw = outputs.float().cpu()
                all_label = labels.float()
                all_fea = feas.float().cpu()
                start_test = False
            else:
                all_output_raw = torch.cat((all_output_raw, outputs.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output_raw)
    _, predict = torch.max(all_output, 1)

    obtained_fea_unk, _, _ = cal_unknown_in_fea(netC, all_fea, easy_idx)

    unknown_idx = identify_unkown_idx(obtained_fea_unk, all_output_raw)

    predict[unknown_idx] = args.class_num

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int), :]
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()
    known_acc = np.mean(acc[:-1])
    HOS = 2 * unknown_acc * known_acc / (unknown_acc + known_acc)
    return known_acc, np.mean(acc), unknown_acc, HOS


def one_batch_loss_calculation(inputs_known, labels_known, inputs_test, netF, netB, netC, bf_fea, NS, idx,
                               unknown_idx_all, polished_initc):
    inputs_test = inputs_test.cuda()
    features_test_F = netF(inputs_test)
    features_test = netB(features_test_F)
    outputs_test = netC(features_test)
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    _, model_predict = torch.max(softmax_out, 1)

    fea_known_F = netF(inputs_known.cuda())
    outputs_known = netC(netB(fea_known_F))
    initc_nei, data_level_pred = obtain_nearest_prototype(features_test, polished_initc)
    initc_nei = torch.from_numpy(initc_nei).cuda()
    labels_known = labels_known.cuda()
    classi_known_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(
        outputs_known, labels_known)

    obtained_fea_unk_batch = norm(np.matmul(features_test.cpu().detach().numpy() - bf_fea, NS), axis=-1)
    obtained_fea_unk_batch = torch.from_numpy(obtained_fea_unk_batch)
    unknown_mask = torch.isin(idx, unknown_idx_all)
    if len(idx.cuda()[~unknown_mask]) != 0:
        p_ent_loss = torch.sum(-softmax_out[~unknown_mask] * torch.log(softmax_out[~unknown_mask] + args.epsilon),
                               dim=1) / np.log(args.class_num)
        LU_certain_loss = torch.exp(torch.mean(p_ent_loss))
        FU_certain_loss = torch.mean(obtained_fea_unk_batch[~unknown_mask], dim=-1)
    else:
        LU_certain_loss, FU_certain_loss = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()

    if len(idx.cuda()[unknown_mask]) != 0:
        n_ent_loss = torch.sum(-softmax_out[unknown_mask] * torch.log(softmax_out[unknown_mask] + args.epsilon),
                               dim=1) / np.log(args.class_num)
        LU_uncertain_loss = torch.exp(args.uncer_para * torch.mean(-n_ent_loss))
        FU_uncertain_loss = -torch.mean(obtained_fea_unk_batch[unknown_mask], dim=-1)
    else:
        LU_uncertain_loss, FU_uncertain_loss = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()

    if len(idx.cuda()[~unknown_mask]) != 0:
        hsic_loss = hsic_normalized(features_test[~unknown_mask], initc_nei[~unknown_mask])
    else:
        hsic_loss = torch.tensor(0.).cuda()

    return classi_known_loss, hsic_loss, LU_certain_loss, LU_uncertain_loss, FU_certain_loss, FU_uncertain_loss


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    for k, v in netC.named_parameters():
        v.requires_grad = False

    base_optimizer = torch.optim.SGD
    optimizer = SAM(param_group, base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    iter_num_update = 0
    best_HOS = 0.0
    best_unknown_acc = 0.
    best_known_acc = 0.
    easy_idx, easy_path = [], []
    polished_initc = np.random.rand(args.class_num, 256)



    while iter_num < max_iter:
        if iter_num % interval_iter == 0:
            iter_num_update += 1
            netF.eval()
            netB.eval()
            netC.eval()
            pred_label_all, feas_F_all, feas_all, output_all_raw, polished_initc, kn_un_labels, easy_idx, easy_path = obtain_cre_domain(
                dset_loaders['test'], netF, netB, netC, args, iter_num_update, easy_idx, easy_path,
            )
            obtained_fea_unk, bf_fea, NS = cal_unknown_in_fea(netC, feas_all, easy_idx)
            unknown_idx_all = identify_unkown_idx(obtained_fea_unk, output_all_raw)
            unknown_idx_all = torch.tensor(unknown_idx_all)

            dset_loaders_known = data_load_list(args, easy_path)
            netF.train()
            netB.train()

        try:
            (inputs_test, _), label_test, idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, _), label_test, idx = next(iter_test)

        try:
            inputs_known, labels_known = next(known_loader_iter)
        except:
            known_loader_iter = iter(dset_loaders_known)
            inputs_known, labels_known = next(known_loader_iter)

        if inputs_test.size(0) == 1:
            continue

        iter_num += 1

        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        classifier_loss = torch.tensor(0.0).cuda()
        classi_known_loss, hsic_loss, LU_certain_loss, LU_uncertain_loss, FU_certain_loss, FU_uncertain_loss = \
            one_batch_loss_calculation(inputs_known, labels_known, inputs_test, netF, netB, netC, bf_fea, NS, idx,
                                       unknown_idx_all, polished_initc)

        classifier_loss += classi_known_loss
        classifier_loss += -args.reg_hsic_para * hsic_loss
        classifier_loss += args.reg_entro_para * LU_certain_loss
        classifier_loss += args.reg_entro_para * LU_uncertain_loss
        classifier_loss += args.reg_resid_para * FU_certain_loss
        classifier_loss += args.reg_resid_para * FU_uncertain_loss



        # -----------------------------------------------------------------------
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        # classifier_loss.backward()
        # optimizer.step()
        classifier_loss.backward()
        optimizer.first_step(zero_grad=True)

        classi_known_loss, hsic_loss, LU_certain_loss, LU_uncertain_loss, FU_certain_loss, FU_uncertain_loss = one_batch_loss_calculation(
            inputs_known, labels_known, inputs_test, netF, netB, netC, bf_fea, NS, idx, unknown_idx_all, polished_initc)

        classifier_loss = torch.tensor(0.0).cuda()
        classifier_loss = classifier_loss + classi_known_loss
        classifier_loss = classifier_loss - args.reg_hsic_para * hsic_loss / (
            (hsic_loss / classi_known_loss).detach())
        classifier_loss = classifier_loss + args.reg_entro_para * LU_certain_loss / (
            (LU_certain_loss / classi_known_loss).detach())
        classifier_loss = classifier_loss + args.reg_entro_para * LU_uncertain_loss / (
            (LU_uncertain_loss / classi_known_loss).detach())
        classifier_loss = classifier_loss + args.reg_resid_para * FU_certain_loss / (
            (FU_certain_loss / classi_known_loss).detach())
        classifier_loss = classifier_loss + args.reg_resid_para * FU_uncertain_loss / (
            (FU_uncertain_loss / classi_known_loss).detach())

        # second forward-backward pass
        classifier_loss.backward()  # Full forward pass
        optimizer.second_step(zero_grad=True)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_os1, acc_os2, acc_unknown, HOS = cal_acc_oda(dset_loaders['test'], netF, netB, netC, easy_idx)

            log_str = '\nTask: {}, Iter:{}/{}, all_acc:{:.2f}, known_acc:{:.2f}, unknown_acc: {:.2f}, ' \
                      'HOS: {:.2f}%'.format(
                args.name, iter_num, max_iter, acc_os2, acc_os1, acc_unknown, HOS)

            if HOS > best_HOS:
                best_HOS = HOS
                best_unknown_acc = acc_unknown
                best_known_acc = acc_os1
                torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F.pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B.pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C.pt"))


            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
    log_str = '\nTask: {}, best_known_acc: {:.2f}, best_unknown_acc: {:.2f}, ' \
              'best_HOS :{:.2f}%'.format(args.name, best_known_acc, best_unknown_acc, best_HOS)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return netF, netB, netC





def cal_unknown_in_fea(netC, feas_all, easy_idx):
    fc_weights = netC.state_dict()['fc.weight_v']
    fc_bias = netC.state_dict()['fc.bias']

    bf_fea = -np.matmul(pinv(fc_weights.cpu().numpy()), fc_bias.cpu().numpy())
    feature_id_val = feas_all[easy_idx]
    if feature_id_val.shape[-1] >= 2048:
        DIM = 1000
    elif feature_id_val.shape[-1] >= 768:
        DIM = 512
    else:
        DIM = feature_id_val.shape[-1] // 2
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_val - bf_fea)

    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    residual = norm(np.matmul(feas_all - bf_fea, NS), axis=-1)

    return residual, bf_fea, NS


 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PDD')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office',
                        choices=['office', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2024, help="random seed")
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)

    parser.add_argument('--bal_par', type=float, default=1.0)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckps/target/')
    parser.add_argument('--output_src', type=str, default='ckps/source/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--confi_nums', type=int, default=10)
    parser.add_argument('--mix_ratio', type=float, default=0)
    parser.add_argument('--uncer_para', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--ent_thres', type=float, default=0.75)
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('--rho', type=float, default=0.05, help="GPU ID")
    parser.add_argument('--clus_para', type=float, default=0.5, help=" ")
    parser.add_argument('--reg_cons', type=float, default=1, help=" ")
    parser.add_argument('--reg_resid_para', type=float, default=1, help=" ")
    parser.add_argument('--reg_entro_para', type=float, default=1, help=" ")
    parser.add_argument('--reg_hsic_para', type=float, default=1, help=" ")
    parser.add_argument('--con_num_growth', type=float, default=0.1, help=" ")
    parser.add_argument('--sigma', type=float, default=0.6,
                        help='')

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        elif args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]
        else:
            args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        if args.da == 'oda':
            args.class_num = 10
            args.class_all = 31
            args.src_classes = [i for i in range(10)]
            args.tar_classes = [i for i in range(31)]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)


