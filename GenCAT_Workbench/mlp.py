import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
import sys

def mlp_load_data(dataset_str, root_path):
    def parse_index_file(filename):
      """Parse index file."""
      index = []
      for line in open(filename):
          index.append(int(line.strip()))
      return index


    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)
    names = ['y', 'tx', 'ty', 'allx', 'ally', 'graph']
      # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(root_path+"/datasets/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
        # with open(root_path+"/gcn-master/gcn/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    y, tx, ty, allx, ally, graph = tuple(objects)
    # x, y, tx, ty, allx, ally, graph = tuple(objects)

    if dataset_str == "cora" or dataset_str == "citeseer" or dataset_str == "pubmed":
        idx_val = range(len(y), len(y)+500)
    else:
        val_idx_reorder = parse_index_file(root_path+"/datasets/ind.{}.val.index".format(dataset_str))
        # test_idx_reorder = parse_index_file(root_path+"/gcn-master/gcn/data/ind.{}.test.index".format(dataset_str))
        val_idx_range = np.sort(val_idx_reorder)
        idx_val = val_idx_range.tolist()


    test_idx_reorder = parse_index_file(root_path+"/datasets/ind.{}.test.index".format(dataset_str))
    # test_idx_reorder = parse_index_file(root_path+"/gcn-master/gcn/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))

    train_mask = sample_mask(list(idx_train)+idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.argmax(labels[train_mask], axis=1)
    y_test = np.argmax(labels[test_mask], axis=1)

    return adj, features.toarray(), y_train, y_test, train_mask, test_mask

def run_mlp(dataset_str, drive_root, iter_count=0):
    adj, features, y_train, y_test, train_mask, test_mask = mlp_load_data(dataset_str,drive_root)
    X_train = features[train_mask]
    X_test = features[test_mask]

    start = time.time()
    clf = MLPClassifier(max_iter=500,validation_fraction=0.4,early_stopping=True).fit(X_train, y_train)
    elapsed_time = time.time() - start
    test_acc = clf.score(X_test, y_test)
    print("accuracy: ", str(test_acc))
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    with open(drive_root+"experimental_results/MLP_"+dataset_str+"_iter"+str(iter_count), 'w') as f:
        f.write(str(test_acc) + "\n" + str(elapsed_time) + "[sec]")
