# %%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
# from models.raw_model import NRGNN
# from models.NRGNN_org import NRGNN
# from models.ema import NRGNN
# from models.label_smooth import NRGNN
from models.NRGNN_V2 import NRGNN

from dataset import Dataset
from Config import parameters

# Training settings
args = parameters()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

np.random.seed(15) # Here the random seed is to split the train/val/test data

# %%
if args.dataset=='dblp':
    from torch_geometric.datasets import CitationFull
    import torch_geometric.utils as utils
    dataset = CitationFull('./data','cora')
    adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9+args.label_rate) * len(labels))]
else:
    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    # print(np.shape(features),features[0])
    # print(features)

    # print('-'*50)
    # print(np.shape(labels), np.unique(labels))
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # print('idx_train: ', np.shape(idx_train))

    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]
    # print('idx_train: ', np.shape(idx_train))
# %% add noise to the labels
noise_labels = labels.copy()

from utils import noisify_with_P
ptb = args.ptb_rate
nclass = labels.max() + 1
train_labels = labels[idx_train]
# print(train_labels, ptb)
noise_y, P = noisify_with_P(train_labels,nclass, ptb, 10, args.noise) 
noise_labels[idx_train] = noise_y


# %% Add edge perturbation
# https://github.com/EnyanDai/RSGNN/blob/main/train_RSGNN.py
# from deeprobust.graph.global_attack import Random
# import random
# random.seed(15)
# attacker = Random()
# n_perturbations = int(args.ptb_rate * (adj.sum()//2))
# print('n_perturbations',n_perturbations)
# attacker.attack(adj, n_perturbations, type='add')
# adj = attacker.modified_adj

# %%
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

esgnn = NRGNN(args,device)
esgnn.fit(features, adj, noise_labels, idx_train, idx_val)

print("=====test set accuracy=======")
esgnn.test(idx_test)
print("===================================")
# %%
