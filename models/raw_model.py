# %%
import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import torch_geometric.utils as utils
import scipy.sparse as sp
from models.GCN import GCN
from utils import accuracy, sparse_mx_to_torch_sparse_tensor


class NRGNN:
    def __init__(self, args, device):

        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.best_model_index = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None

    def fit(self, features, adj, labels, idx_train, idx_val):

        args = self.args

        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)

        self.model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         self_loop=True,
                         dropout=self.args.dropout, device=self.device).to(self.device)



        self.optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            self.train(epoch, features, edge_index, idx_train, idx_val)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

        print("=====validation set accuracy=======")
        self.test(idx_val)
        print("===================================")

    def train(self, epoch, features, edge_index, idx_train, idx_val):
        args = self.args

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        # prediction of the GCN classifier
        predictor_weights = torch.ones([edge_index.shape[1]], device=self.device)
        output = self.model(features, edge_index, predictor_weights)
        pred_model = F.softmax(output, dim=1)

        eps = 1e-8
        pred_model = pred_model.clamp(eps, 1 - eps)


        # loss of GCN classifier
        loss_gcn = F.cross_entropy(output[idx_train], self.labels[idx_train])

        total_loss = loss_gcn
        total_loss.backward()

        self.optimizer.step()

        acc_train = accuracy(output[idx_train].detach(), self.labels[idx_train])

        # Evaluate validation set performance separately,
        self.model.eval()
        output = self.model(features, edge_index, predictor_weights.detach())

        acc_val = accuracy(output[idx_val], self.labels[idx_val])


        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = predictor_weights.detach()
            self.best_model_index = edge_index
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: {:.4f}'.format(self.best_val_acc.item()))

        if args.debug:
            if epoch % 50 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                print('Epoch: {:04d}'.format(epoch + 1),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    def test(self, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        features = self.features
        labels = self.labels


        self.model.eval()
        estimated_weights = self.best_graph
        model_edge_index = self.best_model_index
        output = self.model(features, model_edge_index, estimated_weights)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tGCN classifier results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return float(acc_test)



