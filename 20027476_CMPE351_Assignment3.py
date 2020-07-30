#!/usr/bin/env python
# coding: utf-8

# ### Title: Assignment 3: Analyzing Facebook Large Page-Page Network
# # Author: 200027476 Stefan Robb
# Network analysis on the Facebook Large Page-Page Network

# ## Dataset
# The dataset we are using in this assignment is from Facebook. A detailed description of the dataset could be found at: https://snap.stanford.edu/data/facebook-large-page-page-network.html.
# 
# The whole network is a page-page graph of verified Facebook sites. Nodes represent official Facebook pages while the links are mutual likes between sites. Node features are extracted from the site descriptions that the page owners created to summarize the purpose of the site. This graph was collected through the Facebook Graph API in November 2017 and restricted to pages from 4 categories which are defined by Facebook. These categories are: politicians, governmental organizations, television shows and companies. 

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import csv
import collections
import operator
import random
import node2vec2
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# In[40]:


with open(r"C:\Users\stefa\Desktop\Queens\Fourth Year\CMPE351\Assignments\Assignment 3\companyinfo.csv", 'r', encoding="utf8") as nodecsv: # Open the file                       
    nodereader = csv.reader(nodecsv)
    nodes = [n for n in nodereader][0:]
    
node_names = [n[0] for n in nodes]

with open(r"C:\Users\stefa\Desktop\Queens\Fourth Year\CMPE351\Assignments\Assignment 3\companylink.csv", 'r', encoding="utf8") as edgecsv: # Open the file
    edgereader = csv.reader(edgecsv)
    edges = [e for e in edgereader][0:]
for n in nodes:
    del n[4]
    del n[3]
    
G = nx.Graph()
G.add_nodes_from(node_names)
G.add_edges_from(edges)


# In[3]:


print(nx.info(G))


# In[4]:


pathlengths = list() # compute the shortest path length
for v in G.nodes():
    spl = dict(nx.single_source_shortest_path_length(G, v))
    for p in spl:
        pathlengths.append(spl[p])


# In[5]:


dist = {} # count the distribution of path lengths
for p in pathlengths:
    if p in dist:
        dist[p] += 1
    else:
        dist[p] = 1

plt.bar(list(dist.keys()), dist.values(), color='b') # plot the distribution of the path lengths
plt.show()


# In[6]:


Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0]) # use connected_components to avoid error


# In[7]:


print("Radius: %d" % nx.radius(Gcc))
print("Diameter: %d" % nx.diameter(Gcc))
print("Density: %s" % nx.density(Gcc))


# ### Path Length Histogram

# In[8]:


degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.show()


# ### Log Log Path Length Histogram

# In[9]:


degrees = {}
for d, c in zip(deg,cnt):
    degrees[d] = c
    
degrees_log = {}
for key in degrees:
    degrees_log[key] = np.log10(degrees[key])
    
plt.bar(list(degrees_log.keys()), degrees_log.values(), color='b')
plt.show()


# In[10]:


centrality = nx.degree_centrality(G)
print("Node : Centrality")
for i in range(1,21):
    print(nodes[int(max(centrality.items(), key=operator.itemgetter(1))[0])][2], ": ",           max(centrality.items(), key=operator.itemgetter(1))[1])
    del centrality[max(centrality.items(), key=operator.itemgetter(1))[0]]


# In[11]:


closeness = nx.closeness_centrality(G)
print("Node : Closeness Centrality")
for i in range(1,21):
    print(nodes[int(max(closeness.items(), key=operator.itemgetter(1))[0])][2], ": ",           max(closeness.items(), key=operator.itemgetter(1))[1])
    del closeness[max(closeness.items(), key=operator.itemgetter(1))[0]]


# In[12]:


eigenvector = nx.eigenvector_centrality(G)
print("Node : Eigenvector Centrality")
for i in range(1,21):
    print(nodes[int(max(eigenvector.items(), key=operator.itemgetter(1))[0])][2], ": ",           max(eigenvector.items(), key=operator.itemgetter(1))[1])
    del eigenvector[max(eigenvector.items(), key=operator.itemgetter(1))[0]]


# In[13]:


for n in nodes: # remove duplicate nodes
    if n[2] == "L'OCCITANE en Provence":
        if n[0] == '4094': # leave node with highest centrality
            continue
        try:
            G.remove_node(n[0])
        except:
            'Failed removal'


# In[24]:


eigenvector = nx.eigenvector_centrality(G) # reprint highest eigenvector centrality
print("Node : Eigenvector Centrality")
for i in range(1,21):
    print(nodes[int(max(eigenvector.items(), key=operator.itemgetter(1))[0])][2], ": ",           max(eigenvector.items(), key=operator.itemgetter(1))[1])
    del eigenvector[max(eigenvector.items(), key=operator.itemgetter(1))[0]]


# In[15]:


nx.draw_networkx(G, with_labels=False, node_size=50, node_color='r') # draw the network
plt.show()


# ### edgeSplitter.py

# In[16]:


# Convert sparse matrix to tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# Get normalized adjacency matrix: A_norm
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


# Prepare feed-dict for Tensorflow session
def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.


    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj)  # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu)  # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples)  # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()



    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    counter=0
    for edge in edge_tuples:
        counter+=1
        if counter%100==0:
            print("processed:"+str(counter))
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or                 false_edge in test_edges_false or                 false_edge in val_edges_false:
            continue

        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false,
        # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or                 false_edge in test_edges_false or                 false_edge in val_edges_false or                 false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])




    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false,            val_edges, val_edges_false, test_edges, test_edges_false


# In[17]:


np.random.seed(0) # make sure train-test split is consistent between notebooks
adj_sparse = nx.to_scipy_sparse_matrix(Gcc)

# Perform train-test split, note the current code also provides a validation set. train edges adn test edges represent the true edges, false edges are randomly generated.
adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_sparse, test_frac=.3, val_frac=.1)

g_train = nx.from_scipy_sparse_matrix(adj_train) # new graph object with only non-hidden edges


# In[18]:


# Inspect train/test split
print("Total nodes:", adj_sparse.shape[0])
print("Total edges:", int(adj_sparse.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print("Training edges (positive):", len(train_edges))
print("Training edges (negative):", len(train_edges_false))
print("Validation edges (positive):", len(val_edges))
print("Validation edges (negative):", len(val_edges_false))
print("Test edges (positive):", len(test_edges))
print("Test edges (negative):", len(test_edges_false))


# In[19]:


def get_roc_score(edges_pos, edges_neg, score_matrix):
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]]) # predicted score
        pos.append(adj_sparse[edge[0], edge[1]]) # actual value (1 for positive)
        
    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]]) # predicted score
        neg.append(adj_sparse[edge[0], edge[1]]) # actual value (0 for negative)
        
    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


# In[20]:


adj = nx.adjacency_matrix(Gcc)
# Compute Jaccard Coefficients from g_train
jc_matrix = np.zeros(adj.shape)
for u, v, p in nx.jaccard_coefficient(g_train): # (u, v) = node indices, p = Jaccard coefficient
    jc_matrix[u][v] = p
    jc_matrix[v][u] = p

# Normalize array
jc_matrix = jc_matrix / jc_matrix.max()


# In[21]:


# Calculate ROC AUC and Average Precision
jc_roc, jc_ap = get_roc_score(test_edges, test_edges_false, jc_matrix)

print('Jaccard Coefficient Test ROC score: ', str(jc_roc))
print('Jaccard Coefficient Test AP score: ', str(jc_ap))


# ### node2vec

# In[22]:


#node2vec settings
P = 1 # Return hyperparameter
Q = 1 # In-out hyperparameter
WINDOW_SIZE = 10 # Context size for optimization
NUM_WALKS = 10 # Number of walks per source
WALK_LENGTH = 80 # Length of walk per source
DIMENSIONS = 128 # Embedding dimension
DIRECTED = False # Graph directed/undirected
WORKERS = 8 # Num. parallel workers
ITER = 1 # SGD epochs


# In[25]:


# Preprocessing, generate walks
g_n2v = node2vec2.Graph(g_train, DIRECTED, P, Q) # create node2vec graph instance
g_n2v.preprocess_transition_probs()
walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
walksn = [] # solved an error which left Nonetype values in walks
for val in walks: 
    if val != None : 
        walksn.append(val)
walkslist = [list(map(str, walk)) for walk in walksn]
# Train skip-gram model
model = Word2Vec(walkslist, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

# Store embeddings mapping
emb_mappings = model.wv


# In[28]:


emb_list = []
for node_index in range(0, adj_sparse.shape[0]):
    node_str = str(node_index)
    node_emb = emb_mappings[node_str]
    emb_list.append(node_emb)
emb_matrix = np.vstack(emb_list)


# In[29]:


# Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
def get_edge_embeddings(edge_list):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = emb_matrix[node1]
        emb2 = emb_matrix[node2]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs


# In[30]:


# Train-set edge embeddings
pos_train_edge_embs = get_edge_embeddings(train_edges)
neg_train_edge_embs = get_edge_embeddings(train_edges_false)
train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

# Create train-set edge labels: 1 = real edge, 0 = false edge
train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

# Val-set edge embeddings, labels
pos_val_edge_embs = get_edge_embeddings(val_edges)
neg_val_edge_embs = get_edge_embeddings(val_edges_false)
val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

# Test-set edge embeddings, labels
pos_test_edge_embs = get_edge_embeddings(test_edges)
neg_test_edge_embs = get_edge_embeddings(test_edges_false)
test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

# Create val-set edge labels: 1 = real edge, 0 = false edge
test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])


# In[32]:


# Train logistic regression classifier on train-set edge embeddings
from sklearn.linear_model import LogisticRegression
edge_classifier = LogisticRegression(random_state=0)
edge_classifier.fit(train_edge_embs, train_edge_labels)


# In[33]:


# Predicted edge scores: probability of being of class "1" (real edge)
val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
val_roc = roc_auc_score(val_edge_labels, val_preds)
val_ap = average_precision_score(val_edge_labels, val_preds)


# In[34]:


# Predicted edge scores: probability of being of class "1" (real edge)
test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
test_roc = roc_auc_score(test_edge_labels, test_preds)
test_ap = average_precision_score(test_edge_labels, test_preds)


# In[35]:


print('node2vec Validation ROC score: ', str(val_roc))
print('node2vec Validation AP score: ', str(val_ap))
print('node2vec Test ROC score: ', str(test_roc))
print('node2vec Test AP score: ',str(test_ap))
