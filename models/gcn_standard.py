import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from dgl.nn.pytorch import GraphConv
import dgl
from dgl import DGLGraph

import time

from models.base_model import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout=0.5):
        super(GCN, self).__init__()

        self.g = g

        self.gcn_layer1 = GraphConv(in_feats, n_hidden, activation=activation)

        self.gcn_layer2 = GraphConv(n_hidden, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = self.gcn_layer1(self.g, features)

        h = self.dropout(h)

        h = self.gcn_layer2(self.g, h)

        return h



def get_masks(n,
              main_ids,
              main_labels,
              test_ratio,
              val_ratio,
              seed=1):
    train_mask = np.zeros(n)
    val_mask = np.zeros(n)
    test_mask = np.zeros(n)

    x_dev, x_test, y_dev, y_test = train_test_split(main_ids,
                                                    main_labels,
                                                    stratify=main_labels,
                                                    test_size=test_ratio,
                                                    random_state=seed)

    x_train, x_val, y_train, y_val = train_test_split(x_dev,
                                                      y_dev,
                                                      stratify=y_dev,
                                                      test_size=val_ratio,
                                                      random_state=seed)

    train_mask[x_train] = 1
    val_mask[x_val] = 1
    test_mask[x_test] = 1

    return train_mask, val_mask, test_mask


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask].detach().cpu().numpy()
        _, predicted = torch.max(logits, dim=1)
        predicted = predicted.detach().cpu().numpy()
        f1 = f1_score(labels, predicted, average='micro')
        return f1



class GCN_Model(BaseModel):
    def __init__(self, graph, features, labels=None, dim=80):
        super(GCN_Model, self).__init__(graph, features, dim, labels)

    def learn_embeddings(self):
        features = torch.FloatTensor(self.features)
        labels = torch.LongTensor(self.labels)
        n_classes = len(np.unique(labels))

        mask = []
        for i in range(len(labels)):
           # if self.graph.nodes[i]['is_main']:
            mask.append(1)
            #else:
            #    mask.append(0)
        mask = torch.BoolTensor(mask)

        g = DGLGraph(self.graph)
        g = dgl.transform.add_self_loop(g)
        n_edges = g.number_of_edges()

        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0

        g.ndata['norm'] = norm.unsqueeze(1)

        in_feats = features.shape[1]
        n_hidden = 64
        dropout = 0.5
        lr = 1e-2

        model = GCN(g,
                    in_feats=in_feats,
                    n_hidden=n_hidden,
                    n_classes=n_classes,
                    activation=F.relu,
                    dropout=dropout)

        loss_fcn = torch.nn.CrossEntropyLoss()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20,
                                                            min_lr=1e-10)

        verbose = False
        n_epochs = 100
        best_f1 = -100
        # initialize graph
        dur = []
        for epoch in range(n_epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            logits = model(features)
            loss = loss_fcn(logits[mask], labels[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            f1 = evaluate(model, features, labels, mask)
            scheduler.step(1 - f1)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'best_model.pt')

            if verbose:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | F1 {:.4f} | "
                      "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                    f1, n_edges / np.mean(dur) / 1000))

        model.load_state_dict(torch.load('best_model.pt'))

        self.embeddings = model.gcn_layer1(model.g, features).detach().cpu().numpy()