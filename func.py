import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import scipy.io

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
    com_size_dict = {}
    for com_num, size in enumerate(community_size):
        com_size_dict[com_num] = size
    com_size_dict = dict(sorted(com_size_dict.items(), key=lambda x:x[1],  reverse=True))
#     print(com_size_dict)

    communities = copy.deepcopy(partition)
    partition = []
    for com_num in com_size_dict.keys():
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
    
def cpm_cpd_plot(S,Label):
    k=max(Label)+1
    import seaborn as sns
    class_pref_mean, class_pref_dev = calc_class_features(S,k,Label)
    plt.rcParams["font.size"] = 13
    plt.title("Class preference mean", fontsize=20)
    hm = sns.heatmap(class_pref_mean,annot=True, cmap='hot_r', fmt="1.2f", cbar=False, square=True)
    plt.xlabel("class",size=20)
    plt.ylabel("class",size=20)
    plt.tight_layout()
    plt.show()
    plt.rcParams["font.size"] = 13
    plt.title("Class preference deviation", fontsize=20)
    hm = sns.heatmap(class_pref_dev,annot=True, cmap='hot_r', fmt="1.2f", cbar=False, square=True)
    plt.xlabel("class",size=20)
    plt.ylabel("class",size=20)
    plt.tight_layout()
    plt.show()
    
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
    return S,Label,n,m,k


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
    path = '/Users/seiji/Documents/datasets/factorized-graphs-master/experiments_sigmod20/realData/'
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