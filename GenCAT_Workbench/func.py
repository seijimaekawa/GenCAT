import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import scipy.io

import pickle as pkl
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

import jgraph
import powerlaw

dataset_path = "/content/drive/My Drive/Colab Notebooks/GenCAT/datasets/"


def save_graph(S,X,Label, dataset_str="GenCAT_test", train_val_test_ratio=[0.48,0.32,0.2]):
    train_val_test_num = []
    for _ in train_val_test_ratio:
        train_val_test_num.append(int(_*S.shape[0]))
    # print(train_val_test_num)
    train_num = train_val_test_num[0]
    val_num = train_val_test_num[1]
    test_num = train_val_test_num[2]

    ### GCN default setting ###
    # train_num = 1000
    # val_num = 500
    # test_num = 140

    val_index = list(range(S.shape[0]))[train_num:-test_num]
    test_index = list(range(S.shape[0]))[-test_num:]

    import random
    import copy
    ind_list = list(range(S.shape[0]))
    random.shuffle(ind_list)

    ind_dic = dict()
    rev_dic = dict()
    for i, ind in enumerate(ind_list):
        ind_dic[i] = ind
        rev_dic[ind] = i

    X_shuffle = np.zeros(X.shape)
    for _ in range(len(X)):
        X_shuffle[ind_dic[_],:] = X[_,:]

    X_x = sp.csr_matrix(X_shuffle[:train_num])
    X_tx = sp.csr_matrix(X_shuffle[-test_num:])
    X_allx = sp.csr_matrix(X_shuffle[:-test_num])
    # print(X_x.shape, X_tx.shape, X_allx.shape)

    Label_onehot = np.identity(max(Label)+1)[Label]
    Label_shuffle = np.zeros(Label_onehot.shape)
    for _ in range(len(Label_onehot)):
        Label_shuffle[ind_dic[_],:] = Label_onehot[_,:]

    Label_y = Label_shuffle[:train_num]
    Label_ty = Label_shuffle[-test_num:]
    Label_ally = Label_shuffle[:-test_num]

    nnz = sp.csr_matrix(S).nonzero()
    graph = dict()
    for i in range(S.shape[0]):
        graph[i] = []
    for i in range(len(nnz[0])):
        graph[ind_dic[nnz[0][i]]].append(ind_dic[nnz[1][i]])

    # Attribute
    with open(dataset_path+'ind.'+dataset_str+'.x','wb') as f:
      pkl.dump(X_x,f)
    with open(dataset_path+'ind.'+dataset_str+'.tx','wb') as f:
      pkl.dump(X_tx,f)
    with open(dataset_path+'ind.'+dataset_str+'.allx','wb') as f:
      pkl.dump(X_allx,f)

    # Label
    with open(dataset_path+'ind.'+dataset_str+'.y','wb') as f:
      pkl.dump(Label_y,f)
    with open(dataset_path+'ind.'+dataset_str+'.ty','wb') as f:
      pkl.dump(Label_ty,f)
    with open(dataset_path+'ind.'+dataset_str+'.ally','wb') as f:
      pkl.dump(Label_ally,f)

    # Topology
    with open(dataset_path+'ind.'+dataset_str+'.graph','wb') as f:
      pkl.dump(graph,f)

    # Data split
    with open(dataset_path+'ind.'+dataset_str+'.val.index','w') as f:
      for _ in val_index:
        f.write(str(_)+'\n')
    with open(dataset_path+'ind.'+dataset_str+'.test.index','w') as f:
      for _ in test_index:
        f.write(str(_)+'\n')

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

def loadData(path="./data/cora/", dataset="cora"):
    import scipy.sparse as sp
    def encode_onehot(labels):
        # The classes must be sorted before encoding to enable static class encoding.
        # In other words, make sure the first class always maps to index 0.
        classes = sorted(list(set(labels)))
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        classes_flat = {c: i for i, c in enumerate(classes)}
        labels_flat = []
        for l in labels:
          labels_flat.append(classes_flat[l])
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot, labels_flat
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels, labels_flat = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj)
#     - adj.multiply(adj.T > adj)
    return adj, features, labels_flat

def _load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(dataset_path+"ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(dataset_path+"ind.{}.test.index".format(dataset_str))
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

    return adj, features, np.argmax(labels, axis=1)

def feature_extraction(S,X,Label):
    k = max(Label)+1
    M,D = calc_class_features(S,k,Label)
    H = calc_attr_cor(X, Label)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    class_size = []
    for i in partition:
        class_size.append(len(i))
    class_size = np.array(class_size) / sum(class_size)

    # node degree
    theta = np.zeros(len(Label))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            theta[nnz[0][i]] += 1
            theta[nnz[1][i]] += 1

    return M,D,list(class_size),H,sorted(theta,reverse=True)

def calc_class_features(S,k,Label):
    pref = np.zeros((len(Label),k))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            pref[nnz[0][i]][Label[nnz[1][i]]] += 1
            pref[nnz[1][i]][Label[nnz[0][i]]] += 1
    for i in range(len(Label)):
        pref[i] /= sum(pref[i])
    pref = np.nan_to_num(pref)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    # caluculate average and deviation of class preference
    from statistics import mean, median,variance,stdev
    class_pref_mean = np.zeros((k,k))
    class_pref_dev = np.zeros((k,k))
    for i in range(k):
        pref_tmp = []
        for j in partition[i]:
            pref_tmp.append(pref[j])
        pref_tmp = np.array(pref_tmp).transpose()
        for h in range(k):
            class_pref_mean[i,h] = mean(pref_tmp[h])
            class_pref_dev[i,h] = stdev(pref_tmp[h])
    return class_pref_mean, class_pref_dev

def calc_attr_cor(X, Label):
    k=max(Label)+1
    n=X.shape[0]
    d=X.shape[1]

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    from statistics import mean
    attr_cor = np.zeros((d,k))
    for i in range(k):
        tmp=np.zeros(d)
        for j in partition[i]:
            tmp+=X[j]
        attr_cor[:,i] = tmp/len(partition[i])
    return attr_cor


def S_class_order(S, n, k, Label):
    import scipy.sparse as sp
    import random
    import copy
    partition = []
    k = max(Label)+1
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    for i in range(k):
        random.shuffle(partition[i])

    community_size = []
    for i in range(len(partition)):
        community_size.append(len(list(partition)[i]))
#     print ("community size : " + str(community_size))
    class_size_dict = {}
    for com_num, size in enumerate(community_size):
        class_size_dict[com_num] = size
    class_size_dict = dict(sorted(class_size_dict.items(), key=lambda x:x[1],  reverse=True))
#     print(class_size_dict)

    communities = copy.deepcopy(partition)
    partition = []
    for com_num in class_size_dict.keys():
        for node in list(communities)[com_num]:
               partition.append(node)
    print(len(partition))

    import random
    S_class = sp.dok_matrix((n,n))

    part_dic = {}
    for i in range(n):
        part_dic[partition[i]] = i

    nzs = S.nonzero()
    for i in range(len(nzs[0])):
        S_class[part_dic[nzs[0][i]],part_dic[nzs[1][i]]] = 1

    return S_class

def adj_plot(S,Label):
    n=len(Label)
    k=max(Label)+1
    plot_S = S_class_order(S, n, k, Label)
    plt.rcParams["font.size"] = 20
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("node ID", size = 24)
    ax.set_ylabel("node ID", size = 24)
    ax.spy(plot_S, markersize=.2)
    ticks = []
    for _ in range(int(n/1000)+1):
        if len(Label) > 5000 and _ % 2 == 0:
            continue
        ticks.append(_*1000)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    plt.show()

def cpm_cpd_plot(S,Label,wide=None,cpm=True,cpd=True):
    k=max(Label)+1
    import seaborn as sns
    class_pref_mean, class_pref_dev = calc_class_features(S,k,Label)
    if wide == None:
      if cpm:
        plt.rcParams["font.size"] = 13
        plt.title("Class preference mean", fontsize=20)
        hm = sns.heatmap(class_pref_mean,annot=True, cmap='hot_r', fmt="1.2f", cbar=False, square=True)
        plt.xlabel("class",size=20)
        plt.ylabel("class",size=20)
        plt.tight_layout()
        plt.show()
      if cpd:
        plt.rcParams["font.size"] = 13
        plt.title("Class preference deviation", fontsize=20)
        hm = sns.heatmap(class_pref_dev,annot=True, cmap='hot_r', fmt="1.2f", cbar=False, square=True)
        plt.xlabel("class",size=20)
        plt.ylabel("class",size=20)
        plt.tight_layout()
        plt.show()
    else:
      plt.subplot(1,3,1)
      plt.rcParams["font.size"] = 10
      plt.title("Class preference mean", fontsize=14)
      hm = sns.heatmap(class_pref_mean,annot=True, cmap='hot_r', fmt="1.2f", cbar=False, square=True)
      plt.xlabel("class",size=20)
      plt.ylabel("class",size=20)
      # plt.tight_layout()

      plt.subplot(1,3,3)
      plt.rcParams["font.size"] = 10
      plt.title("Class preference deviation", fontsize=14)
      hm = sns.heatmap(class_pref_dev,annot=True, cmap='hot_r', fmt="1.2f", cbar=False, square=True)
      plt.xlabel("class",size=20)
      plt.ylabel("class",size=20)
      # plt.tight_layout()
      plt.show()
    return class_pref_mean, class_pref_dev

def att_plot(X,Label,tag):
    k=max(Label)+1
    plt.rcParams["font.size"] = 21
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1,1,1)
    colors = ['red','blue','green','purple','gold','brown','c','m','k','plum','yellow','pink','maroon','teal','tomato']
    markers = ['.',',','v','^']

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    count = 1
    for i in partition:
        tmp_ver = []
        tmp_hor = []
        for j in i:
            tmp_ver.append(X[j,0])
            tmp_hor.append(X[j,1])
#         ax.scatter(tmp_ver,tmp_hor, c=colors[count],label=count, s=6, marker=markers[count])
        ax.scatter(tmp_ver,tmp_hor, c=colors[count-1],label=count, s=0.05)
        count+=1
        if count == 5: # how many classes do you want to plot?
            break

    plt.xlabel("attribute1", size=32)
    plt.ylabel("attribute2", size=32)
    plt.legend(bbox_to_anchor=(0.45, 1.0), loc='lower center', borderaxespad=1., ncol=4 , markerscale=10., scatterpoints=1, fontsize=18,title='class').get_title().set_fontsize(30)
    plt.tight_layout()
    plt.show()




def load_data(file_name):
    if file_name[-3:] == 'npz':
        return load_npz(file_name)
    elif file_name[-3:] == 'mat':
        return load_mat(file_name)
    else:
        return load_csv(file_name)

def load_npz(file_name):
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)['arr_0'].item()
        S = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])
        if 'attr_data' in loader:
            _X_obs = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            _X_obs = None
        Label = loader.get('labels')
    S= S + S.T
    S[S > 1] = 1
    lcc = largest_connected_components(S)
    S = S[lcc,:][:,lcc]
    Label = Label[lcc]
    n = S.shape[0]
    k = len(set(Label))
    for i in range(n):
        S[i,i] = 0
    nonzeros = S.nonzero()
    m = int(len(nonzeros[0])/2)
    print ("number of nodes : " + str(n))
    print ("number of edges : " + str(m))
    print ("number of classes : " + str(len(set(Label))))
    return S,_X_obs,Label,n,m,k


def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def load_mat(path): # switch for two form of file
    if "mat" in path:
        print ("mat")
        S,Label,A = for_mat(path)
        nnz = S.nonzero()
        return S,Label,S.shape[0],int(len(nnz[0])/2),len(set(Label))

def for_mat(path):
    mat_contents = scipy.io.loadmat(path)
#     print(mat_contents)
    G = mat_contents["S"]
    X = mat_contents["X"]
    Label =np.ndarray.flatten(mat_contents["C"])
    node_size = G.shape[0]
    att_size = X.shape[1]
    S = np.zeros((node_size,node_size))
    if type(X) != np.ndarray:
        A = X.toarray()
    else:
        A = X
    #fill the adjacency matrix and attribute matrix
    nonzeros = G.nonzero()
    print ("no.nodes: " + str(node_size))
    print ("no.attributes: " + str(att_size))
    edgecount=0
    for i in range(len(nonzeros[0])):
        S[nonzeros[0][i],nonzeros[1][i]] = 1
        S[nonzeros[1][i],nonzeros[0][i]] = 1
    # erase diagonal element
    diag = 0
    for i in range(node_size):
#         diag += S[i,i]
        S[i,i] = 0
    return S, Label, A

def load_csv(file_name):
    path = ''
    with open(path+file_name+'-neighbors.csv',mode='r') as f:
        edges = f.read().split('\n')[:-1]
    for i, edge in enumerate(edges):
        edges[i] = edge.split(',')
    with open(path+file_name+'-classes.csv',mode='r') as f:
        classes = f.read().split('\n')[:-1]
    n = len(classes)
    C = np.zeros(n,dtype=int)
    for tmp in classes:
        i,c_i = tmp.split(',')
        C[int(i)] = int(c_i)
    S = sp.lil_matrix((n,n),dtype=int)
    for i,j in edges:
        S[int(i),int(j)] = 1
        S[int(j),int(i)] = 1
    S = S.tocsr()
    return S, C, n, S.sum(), len(set(C))


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC
    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    The wedge count.
    """
    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0).flatten()
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.
    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = jgraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0).flatten()
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0).flatten()
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
                                                                                                               n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.
    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0).flatten()
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er


def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_graph_statistics(A_in, Z_obs=None):
    """
    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Z_obs: np.matrix [N, K], where K is the number of classes.
          Matrix whose rows are one-hot vectors indicating the class membership of the respective node.
    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """
    try:
      A = A_in.copy().toarray()
    except Exception:
      A = A_in.copy()

    assert((A == A.T).all())
    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['degree_max'] = d_max
    statistics['degree_min'] = d_min
    statistics['degree_mean'] = d_mean

    # node number & edger number
    statistics['node_num'] = A_graph.number_of_nodes()
    statistics['edge_num'] = A_graph.number_of_edges()

    # largest connected component
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]
    # wedge count
    # statistics['wedge_count'] = statistics_wedge_count(A)

    # claw count
    # statistics['claw_count'] = statistics_claw_count(A)

    # triangle count
    # statistics['triangle_count'] = statistics_triangle_count(A)

    # Square count
    # statistics['square_count'] = statistics_square_count(A)

    # power law exponent
    # statistics['power_law_exp'] = statistics_power_law_alpha(A)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)

    # Relative edge distribution entropy
    # statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)

    # Assortativity
    # statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)

    # Clustering coefficient
    # statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / (statistics['claw_count']+1)

    # Number of connected components
    statistics['n_components'] = connected_components(A)[0]

    # if Z_obs is not None:
    #     # inter- and intra-community density
    #     intra, inter = statistics_cluster_props(A, Z_obs)
    #     statistics['intra_community_density'] = intra
    #     statistics['inter_community_density'] = inter

    statistics['cpl'] = statistics_compute_cpl(A)

    return statistics
