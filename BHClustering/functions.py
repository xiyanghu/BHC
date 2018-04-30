from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, vectorize, float64, int64
import seaborn as sns
import pandas as pd
import math 


class mynode():
    '''
    Class of node
    
    A node have:
      self.l: a left child
      self.r: a right child
      self.data: the data
      self.cluster: which cluster it belongs to
      self.n: number of nodes of the subtree of the node
      self.d: d value
      self.pi: pi value
      self.index: index of the node
      self.parent: index of the parent
    '''
    def __init__(self, left, right, data, cluster, alpha, index = None, parent = None):
        self.l = left
        self.r = right
        self.data = data
        self.cluster = cluster
            
        if(left == None):
            self.n = 1
            self.d = alpha
        else:
            self.n = left.n + right.n
            self.d = alpha*special.gamma(self.n) + left.d*right.d
                
        self.pi = alpha*special.gamma(self.n)/self.d
        self.index = index
        self.parent = parent

        
def p_dk_h1k(data, alpha):
    '''
    To calculate the probability under hypothesis 1
    
    Input：
    data is an array
    alpha is the learning hyperparamter mentioned in the essay
    '''
    
    N=data.shape[0]
    k=data.shape[1]
    if(k==1):
        alpha=[0.9,0.1,0.5]
    prod = [special.loggamma(np.sum(data[i, :])+1)-np.sum(special.loggamma(data[i, :]+1)) for i in range(N)]
    multiplier1 = np.sum(prod)
    
    prod2 = [special.loggamma(alpha[j]+np.sum(data[:, j]))-special.loggamma(alpha[j]) for j in range(k)]
    multiplier2 = np.sum(prod2)
    
    logprob = multiplier1 + multiplier2 + special.loggamma(np.sum(alpha))-special.loggamma(np.sum(data) + np.sum(alpha))
    prob = np.real(np.exp(logprob))
    
    return prob


def pik(mynode, alpha):
    '''
     To calculate the pi_k value of the node
    
    Input：
      node
      alpha
    '''
    
    if mynode.l == None:
        dk = alpha
    else:
        dk = alpha*special.gamma(mynode.n) + mynode.l.d*mynode.r.d
    
    mynode.d = dk
    pi_k = alpha*special.gamma(mynode.n)/dk
    mynode.pi = pi_k
    
    return pi_k


def p_dk_tk(mynode, alpha=0.5):
    '''
    To calculate the P(D_k|T_k) value of the node
    
    Input：
      node
      alpha
    '''
    p = p_dk_h1k(mynode.data, np.repeat(alpha, mynode.data.shape[1]))
    pi = pik(mynode, alpha)
    if mynode.l == None:
        return  pi * p
    else:
        return  pi * p + (1-pi) * p_dk_tk(mynode.l, alpha) * p_dk_tk(mynode.r, alpha)

    
def find_max_rk(mynode_list, alpha, dim):
    '''
    To find the pair i and j with the largest r_k value
    
    Input:
      nodelist
      alpha
      dimension of the data
      
    Return:
      the index of the pair: i_max, j_max
      the maximum r_k: rk_max
    '''
    
    rk_max = -10000
    i_max = 0
    j_max = 1
    
    # find the pair i and j with the highest probability of the merged hypothesis
    for i in range(len(mynode_list)):
        for j in range(i+1, len(mynode_list)):
            
            # concatenate the data of i and j
            newdata = np.concatenate((mynode_list[i].data, mynode_list[j].data), axis = 0)
                
            # creat a new mynode with i and j as its children
            mynode_new = mynode(mynode_list[i], mynode_list[j], newdata,
                                min(mynode_list[i].cluster,mynode_list[j].cluster), alpha)
            
            # calculate the r_k value
            rk = pik(mynode_new, alpha) * p_dk_h1k(mynode_new.data, np.repeat(alpha, dim)) / p_dk_tk(mynode_new, alpha)
            
            # record the largest r_k
            if (rk>rk_max):
                rk_max = rk
                i_max = i
                j_max = j
                
    return i_max, j_max, rk_max



def set_cluster(root,clusternum):
    '''
    Set the cluster number of all nodes in the subtree to clusternum
    '''
    queue = []

    queue.append(root)
    while queue:
        newNode = queue.pop(0)
        newNode.cluster = clusternum
        
        if newNode.l != None:
            queue.append(newNode.l)
        if newNode.r != None:
            queue.append(newNode.r)

    return



def bhc(data, alpha = 0.5, r_thres = 0.91):
    '''
    Bayesian Hierarchical Clustering
    
    Input：
      data 
      alpha
      threshold of r_k
    
    The tree can be cut at points where r_k<r_thres
    
    The function will return tree[0], assignments
      tree[0]: The root of the tree
      assignments: The cluster numbers of each merge
    '''
    
    tree = []
        
    num = data.shape[0]
    # initialize
    for i in range(num):
        tree.append(mynode(None, None, np.array([data[i,:]]), i, alpha, i, None))
        
    # number of clusters
    c = data.shape[0]
    
    # list for recording the cluster number
    assignment = [i for i in range(num)]
    assignments = [list(assignment)]
    
    while (c>1):        
        
        # determine which pair to merge (pair i,j with the largest r_k)
        tree_i_max, tree_j_max, tree_rk_max = find_max_rk(tree, alpha, data.shape[1])
        
        # the smaller cluster number
        tree_clusternum = min(tree[tree_i_max].cluster,tree[tree_j_max].cluster)
        # the larger cluster number
        max_clusternum = max(tree[tree_i_max].cluster,tree[tree_j_max].cluster)
        
        # set cluster numbers to the smaller one
        set_cluster(tree[tree_i_max],tree_clusternum)
        set_cluster(tree[tree_j_max],tree_clusternum)
        
        # creat a newnode with i,j as children
        tree_mynode_new_max = mynode(tree[tree_i_max], tree[tree_j_max], np.concatenate((tree[tree_i_max].data, tree[tree_j_max].data), axis = 0), 
                                tree_clusternum, alpha, index = num+data.shape[0]-c)
        
        # set the parent of i,j
        tree[tree_i_max].parent = tree_mynode_new_max.index
        tree[tree_j_max].parent = tree_mynode_new_max.index
        
        # construct the tree from leaves to the roots
        tree =  tree[:tree_i_max] + tree[(tree_i_max+1):tree_j_max] + tree[(tree_j_max+1):]
        tree = [tree_mynode_new_max] + tree
        
        c = c - 1
        
        # set the cluster number, record the cluster numbers of each merge
        for i, k in enumerate(assignment):
            if k == max_clusternum:
                assignment[i] = tree_clusternum
        assignments.append(list(assignment))
            
    return tree[0], assignments



def draw_dendrogram(root):
    '''
    Manually draw a dendrogram of the tree from root
    '''
    
    from anytree import Node, RenderTree
    
    # a queue to go through all the nodes in the subtree
    queue = []

    queue.append(root)
    while queue:
        newNode = queue.pop(0)
        
        # name the node as "D" + index
        ind = "D"+str(newNode.index)
        par = "D"+str(newNode.parent)
        if(newNode.parent == None):
            exec(ind + " = Node(\"" + ind + "\")")
        else:
            exec(ind + " = Node(\"" + ind + "\", parent=" + par + ")")
                        
        if newNode.l != None:
            queue.append(newNode.l)
        if newNode.r != None:
            queue.append(newNode.r)
    
    # draw the dendrogram
    exec("print(RenderTree(" + "D"+str(root.index) + "))")
    
    return

