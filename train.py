import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from models.dgi import DGI, MultiClassifier
from models.subgi import SubGI
from IPython import embed
import scipy.sparse as sp
from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from collections import defaultdict
from sklearn.manifold import SpectralEmbedding
import argparse


def extract_node_degree(graph, max_degree = 32):
    """one-hot node degree"""
    features = th.zeros([graph.number_of_nodes(), max_degree])
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features

##### build graph
def build_graph(file_path, label_path):
    graph = nx.Graph()
    with open(file_path) as IN:
        for line in IN:
            tmp = line.strip().split()
            graph.add_edge(int(tmp[0]), int(tmp[1]))
    labels = dict()
    with open(label_path) as IN:
        IN.readline()
        for line in IN:
            tmp = line.strip().split(' ')
            labels[int(tmp[0])] = int(tmp[1])    

    node_mapping = defaultdict(int)
    relabels = []
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
        relabels.append(labels[node])
    
    assert len(node_mapping) == len(labels)
    new_g = DGLGraph()
    new_g.add_nodes(len(node_mapping))
    for i in range(len(node_mapping)):
        new_g.add_edge(i, i)
    for edge in graph.edges():
        new_g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        new_g.add_edge(node_mapping[edge[1]], node_mapping[edge[0]])

    return new_g, relabels

def get_model(args, graph, in_feats:int):
    if args.model_type == 0:
        dgi = DGI(graph,
                  in_feats,
                  args.n_hidden,
                  args.n_layers,
                  nn.PReLU(args.n_hidden),
                  args.dropout)
    elif args.model_type == 2:
        dgi = SubGI(graph,
                    in_feats,
                    args.n_hidden,
                    args.n_layers,
                    nn.PReLU(args.n_hidden),
                    args.dropout,
                    args.model_id)

    return dgi


def random_train_test_mask(labels, valid_mask = None, train_ratio=0.8):
    train_mask = th.zeros(labels.shape, dtype=th.bool)
    test_mask = th.ones(labels.shape, dtype=th.bool)
    
    num_train = int(labels.shape[0] * train_ratio)
    all_node_index = list(range(labels.shape[0]))
    np.random.shuffle(all_node_index)

    train_mask[all_node_index[:num_train]] = 1
    test_mask[all_node_index[:num_train]] = 0
    if valid_mask is not None:
        train_mask *= valid_mask
        test_mask *= valid_mask
    return train_mask, test_mask

def evaluate(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


##### Train with source graph
def train_src(args, file_path, label_path):
    """Prepare source"""
    src_graph, src_labels = build_graph(file_path, label_path)
    src_labels = th.LongTensor(src_labels)
    features = extract_node_degree(src_graph)
    src_graph.readonly()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        features = features.cuda()
        src_labels = src_labels.cuda()

    n_classes = src_labels.max().item() + 1
    in_feats = features.shape[1]
    dgi = get_model(args, src_graph, in_feats)

    if cuda:
        dgi.cuda()
    dgi_optimizer = th.optim.Adam(dgi.parameters(),
                                        lr=args.dgi_lr,
                                        weight_decay=args.weight_decay)
    cnt_wait = 0
    best = 1e9
    src_graph.ndata['features'] = features

    train_sampler = dgl.contrib.sampling.NeighborSampler(src_graph, 256, 5,
                                                         neighbor_type='in', num_workers=8,
                                                         add_self_loop=False,
                                                         num_hops=args.n_layers + 1, shuffle=True)

    """Train GNN"""    
    for i in range(args.n_dgi_epochs):
        dgi.train()
        loss = 0.0
        
        # EGI mode
        if args.model_type == 2:
            for nf in train_sampler:
                dgi_optimizer.zero_grad()
                l = dgi(features, nf)
                l.backward()
                loss += l
                dgi_optimizer.step()        
        # DGI mode
        elif args.model_type == 0:
            dgi_optimizer.zero_grad()
            loss = dgi(features)
            loss.backward()
            dgi_optimizer.step()
        
        if loss < best:
            print('best')
            best = loss
            cnt_wait = 0
            th.save(dgi.state_dict(), 'best_classification_{}.pkl'.format(args.model_type))
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    """Train Task"""
    # create classifier model
    classifier = MultiClassifier(args.n_hidden, n_classes)
    if cuda:
        classifier.cuda()

    classifier_optimizer = th.optim.Adam(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)

    dgi.load_state_dict(th.load('best_classification_{}.pkl'.format(args.model_type)))

    # extract embeddings
    with th.no_grad():
        if args.model_type == 2:
            embeds = dgi.encoder(features, corrupt=False)
        elif args.model_type == 0:
            embeds = dgi.encoder(features)
    
    embeds = embeds.detach()
    print(embeds.shape)

    train_mask, test_mask = random_train_test_mask(src_labels)
    if hasattr(th, 'BoolTensor'):
        train_mask = th.BoolTensor(train_mask)
        test_mask = th.BoolTensor(test_mask)
    else:
        train_mask = th.ByteTensor(train_mask)
        test_mask = th.ByteTensor(test_mask)

    dur = []
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[train_mask], src_labels[train_mask])
        loss.backward()
        classifier_optimizer.step()

    acc = evaluate(classifier, embeds, src_labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))

    return 

def train_dst(args, file_path, label_path):
    """Prepare source"""
    dst_graph, dst_labels = build_graph(file_path, label_path)
    dst_labels = th.LongTensor(dst_labels)
    features = extract_node_degree(dst_graph)
    dst_graph.readonly()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        features = features.cuda()
        dst_labels = dst_labels.cuda()

    n_classes = dst_labels.max().item() + 1
    in_feats = features.shape[1]
    dgi = get_model(args, dst_graph, in_feats)

    """Train Task"""
    # create classifier model
    classifier = MultiClassifier(args.n_hidden, n_classes)
    if cuda:
        classifier.cuda()

    classifier_optimizer = th.optim.Adam(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)

    dgi.load_state_dict(th.load('best_classification_{}.pkl'.format(args.model_type)))

    # extract embeddings
    with th.no_grad():
        if args.model_type == 2:
            embeds = dgi.encoder(features, corrupt=False)
        elif args.model_type == 0:
            embeds = dgi.encoder(features)
    
    embeds = embeds.detach()
    print(embeds.shape)

    train_mask, test_mask = random_train_test_mask(dst_labels)
    if hasattr(th, 'BoolTensor'):
        train_mask = th.BoolTensor(train_mask)
        test_mask = th.BoolTensor(test_mask)
    else:
        train_mask = th.ByteTensor(train_mask)
        test_mask = th.ByteTensor(test_mask)

    dur = []
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[train_mask], dst_labels[train_mask])
        loss.backward()
        classifier_optimizer.step()

    acc = evaluate(classifier, embeds, dst_labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DGI')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--dgi-lr", type=float, default=1e-2,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=1e-2,
                        help="classifier learning rate")
    parser.add_argument("--n-dgi-epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=0,
                        help="early stop patience condition")
    parser.add_argument("--model", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--model-type", type=int, default=2,
                    help="graph self-loop (default=False)")
    parser.add_argument("--graph-type", type=str, default="DD",
                    help="graph self-loop (default=False)")
    parser.add_argument("--file-path", type=str,
                        help="graph path")
    parser.add_argument("--label-path", type=str,
                        help="label path")
    parser.add_argument("--model-id", type=int, default=2,
                    help="[0, 1, 2, 3]")

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    # train_src(args, 'data/europe-airports.edgelist', 'data/labels-europe-airports.txt')
    train_dst(args, 'data/europe-airports.edgelist', 'data/labels-europe-airports.txt')
    train_dst(args, 'data/usa-airports.edgelist', 'data/labels-usa-airports.txt')
    train_dst(args, 'data/brazil-airports.edgelist', 'data/labels-brazil-airports.txt')