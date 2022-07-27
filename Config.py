import torch

class parameters():
    def __init__(self):
        super(parameters, self).__init__()
        self.debug = True
        self.cuda = True
        self.seed = 13
        self.weight_decay = 5e-4 # 'Weight decay (L2 loss on parameters).'
        self.hidden = 16
        self.edge_hidden = 64
        self.dropout = 0.5
        self.dataset = "dblp" # choices=['cora', 'citeseer','pubmed','dblp']
        self.ptb_rate = 0.2 # noise ptb_rate
        self.epochs = 1
        self.lr = 0.001
        self.alpha = 0.03 # weight of loss of edge predictor
        self.beta = 1 # weight of the loss on pseudo labels
        self.t_small = 0.1 # threshold of eliminating the edges
        self.p_u = 0.8 # threshold of adding pseudo labels'
        self.n_p = 100 # number of positive pairs per node
        self.n_n = 100 # number of negitive pairs per node
        self.label_rate = 0.05 # rate of labeled data
        self.noise = 'pair' # choices=['uniform', 'pair'], help='type of noises'
