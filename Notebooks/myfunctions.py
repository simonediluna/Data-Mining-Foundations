#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import heapq as hp
from kneed import KneeLocator
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist, squareform
from itertools import combinations
from copy import deepcopy

# Data Quality & Outliers
def z_normalization(df, ddof = 1):
    return (df - df.mean()) / df.std(ddof = ddof)

def skew(serie): # misura della distorsione di un attributo
    n = serie.count()
    adj_fact = np.sqrt(n*(n-1)) / (n-2)
    z_score = (serie - serie.mean()) / serie.std(ddof = 0)
    return adj_fact * (z_score**3).mean()

def skew_df(df): # misura della distorsione di tutti gli attributi di un dataframe
    skew_dict = {}
    for attributo in df.columns:
        skew_dict[attributo] = [format(skew(df[attributo]), '.2f')]        
    dataframe = pd.DataFrame(skew_dict, index = ['skew'])
    return dataframe

def kurt(serie, unbiased = False): # curtosi di un attributo
    z_score = (serie - serie.mean()) / serie.std(ddof = 0)
    kurt_biased = (z_score**4).mean() - 3 
    if not unbiased:
        return kurt_biased
    n = serie.count()
    adj_fact = (n - 1) / ((n - 2) * (n - 3))
    return adj_fact * ((n + 1) * kurt_biased + 6)

def kurt_df(df, unbiased = False): # curtosi di tutti gli attributi di un dataframe
    kurt_dict = {}
    if not unbiased:
        for attributo in df.columns:
            kurt_dict[attributo] = [format(kurt(df[attributo]), '.2f')]
    else:
        for attributo in df.columns:
            kurt_dict[attributo] = [format(kurt(df[attributo], unbiased = True), '.2f')]        
    dataframe = pd.DataFrame(kurt_dict, index = ['curtosi'])
    return dataframe  

def outlier_detector(df:'DataFrame', k:'= 1.5 by default' = 1.5, perc:'=False by default' = False): # trova gli outlier    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3-Q1
    outliers = pd.DataFrame(df[(df < Q1 - k * IQR) | (df > Q3 + k * IQR)].count(), columns = ['outliers']).T  
    if not perc:
        return outliers
    attributes = df.columns.tolist()
    formats = ["{:.2%}"] * df.shape[1]
    formatted = dict(zip(attributes, formats))
    outliers = outliers / df.count()
    return outliers.style.format(formatted)
    
def remove_outliers(df:'dataframe'):
    cleaned_df = df.copy()
    Q1 = cleaned_df.quantile(0.25)
    Q3 = cleaned_df.quantile(0.75)
    IQR = Q3-Q1
    cleaned_df = cleaned_df[(cleaned_df >= Q1-1.5*IQR) & (cleaned_df <= Q3+1.5*IQR)].dropna()
    return cleaned_df

# Regression
def adj_r2(r2:'r2_score', n:'sample size', k:'number of regressors (intercept icluded)'):
    fact = (n-1) / (n-k)
    return 1 - (1-r2) * fact

# Principal Components
def my_pca(df:'dataframe', n_comp:'pc numbers' = None)->'df ridotto':
    n_comp = df.shape[1] if n_comp == None else n_comp
    pca = PCA(n_components = n_comp).fit(df)
    projected = pca.transform(df)  
    idx = df.index
    projected = pd.DataFrame(projected, columns = ['pc'+ str(i) for i in range(1,n_comp+1)], index = idx)
    return projected

# Clustering    
def multi_sse(df:'dataframe', k:'numero di cluster'=10, init:'' = 'k-means++', n_init:'' = 10, max_iter:'' = 300, tol:''= 0.0001, random_state:'' = None)->'lsita di SSE e modelli kmeans':
    if df.ndim == 1:
        X = df.values.reshape(-1,1)
    else:
        X = df.values
    SSE_scores = []
    kmeans_models = []
    for num_clust in range(2,k+1):
        model = KMeans(n_clusters = num_clust, init = init, n_init = n_init, max_iter=max_iter, tol=tol, random_state = random_state).fit(X)
        SSE_scores.append(model.inertia_)
        kmeans_models.append(model)
    return SSE_scores, kmeans_models

def multi_silh_scores(proximity_mtx:'proximity matrix', kmeans_models:'lista di modelli di kmeans')->'lista di silhouette score, uno per ogni modello':
    silh_scores = []
    for model in kmeans_models:
        score = silhouette_score(proximity_mtx, model.labels_)
        silh_scores.append(score)
    return silh_scores

def plot_elbow(SSE_scores:'lista di SSE', kneelocator:'find the knee point' = True, legend:'text' = '', linestyle:'' = '--o'):
    k = len(SSE_scores)
    k += 2
    plt.plot(range(2,k), SSE_scores, linestyle, label = legend)
    plt.xticks(range(2,k,2))
    if kneelocator:
        x_elbow = KneeLocator(range(k-2), SSE_scores, curve='convex', direction='decreasing').knee # KneeLocator requires k > 6
        y_elbow = SSE_scores[x_elbow]
        x_elbow += 2
        plt.plot(x_elbow, y_elbow, 'ks', ms = 8, label='_nolegend_')
        plt.annotate(f'({x_elbow}, {y_elbow:.0f})', (x_elbow*1.05, y_elbow*1.05))
    plt.xlabel('Number of clusters')
    plt.ylabel('Total SSE')

def plot_silhouette(silh_scores:'lista di silhouette score, uno per ogni modello', annotate:'bool' = True, legend:'text' = '', linestyle:'' = '--o'):
    k = len(silh_scores)+2
    plt.plot(range(2,k), silh_scores, linestyle, label = legend)
    if annotate:
        x_max_score = np.argmax(silh_scores)
        y_max_score = silh_scores[x_max_score]
        x_max_score += 2    
        plt.plot(x_max_score, y_max_score, 'ks', ms = 8, label='_nolegend_')
        plt.annotate(f'({x_max_score}, {y_max_score:.3f})', ((x_max_score)*1.3, y_max_score))
        plt.ylim(top = y_max_score * 1.05)
    plt.xticks(range(2,k,2))
    plt.xlabel('Number of clusters')
    plt.ylabel('Average silhouette score')

def parallel_centroids(df:'dataframe', model:'kmeans model', title:'bool' = False):
    X = df.values
    labels = model.labels_
    centroids = model.cluster_centers_
    k = len(centroids)
    cols = df.columns.tolist()
    clusters = ['C'+str(i) for i in range(1,k+1)]
    for i, C in enumerate(centroids):
        plt.plot(C, marker='o', label = clusters[i]) 
    if title:
        plt.title(f'{k} clusters')
    plt.xticks(range(len(cols)), cols)
    plt.yticks()
    plt.xlim([0,len(cols)-1])
    plt.grid(visible=True, axis='x')
    
def intracluster_SSEs(X:'dataframe o ndarray', model:'kmeans model', approx:'' = 2)->'lista di SSE, uno per ogni cluster':
    if not isinstance(X, np.ndarray): 
        X = X.values 
    centroids = model.cluster_centers_
    intracluster_SSE = []
    for cluster_id in np.unique(model.labels_):
        sub_X = X[model.labels_ == cluster_id,:]
        sse = SSE(sub_X, centroid = centroids[cluster_id])
        intracluster_SSE.append(sse.round(approx))
    return np.array(intracluster_SSE)

def wcss_barplot(intracluster_SSEs:'lista di SSE, uno per ogni cluster'):
    my_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    x = list(range(intracluster_SSEs.size))
    plt.bar(x, intracluster_SSEs, color = sns.color_palette())
    plt.xlabel('Clusters')
    plt.ylabel('Intra-cluster SSE')
    ytks = np.linspace(0, intracluster_SSEs.max(),5).round(0)
    plt.yticks(ytks)
    plt.xticks(x, labels = ['C'+ str(i+1) for i in x])
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), my_colors):
        ticklabel.set_color(tickcolor)

def cluster_densities(X:'df ndarray', labels:'1darray'):
    '''
    Ritorna un dataframe che associa ad ogni cluster la sua densità. Per ciascun 
    cluster, la densità è computata come la media delle norme L2.
    '''
    clustering = {lab:X[labels == lab,:] for lab in np.unique(labels)}
    densities = {lab:[] for lab in labels}
    for lab, C in clustering.items():
        density = np.sqrt(((C - C.mean(axis=0))**2).sum(axis=1)).mean()
        densities[lab].append(density)
    output = pd.DataFrame(densities, index = ['cluster densities'])
    n_clust = len(np.unique(labels))
    output = output.reindex(range(n_clust), axis=1)
    output.columns = ['C'+str(lab) for lab in range(n_clust)]
    return output

def SSE(X:'cluster ndarray', centroid:'' = None):
    '''
    Computa l'SSE di uno specifico cluster, X. Se il centroide non viene
    dato in input, si considera l'array delle medie delle dimensioni di X.
    '''
    if centroid is None:
        return (((X - X.mean(axis=0))**2).sum(axis=1)).sum() 
    return (((X - centroid)**2).sum(axis=1)).sum()

# total SSE per i modelli non prototype    
def total_SSE(X, labels):
    total_sse = 0
    for lab in np.unique(labels):
        cluster = X[labels == lab, :]
        sse = SSE(cluster)
        total_sse += sse
    return total_sse
    
def hier_kmeans(X:'dataset ndarray', labels:'labels', n_clust:'number of cluster' = 4, method:'link method' = 'ward')->'4 output: clustering, new_labels, new_X, merge_history':
    
    def proximity(mtx, method = 'average'):
        if method == 'single':
            return mtx.min()
        elif method == 'complete':
            return mtx.max()
        elif method == 'average':
            return mtx.mean()       
    
    def recompone(clustering):
        new_labels = []
        new_X = None
        for lab, cluster in clustering.items():
            new_labels += [lab] * cluster.shape[0]
            if new_X is None:
                new_X = cluster
            else:
                new_X = np.vstack((new_X, cluster))
        # new_X = np.concatenate(list(clustering.values()), axis=0)
        return np.array(new_labels), new_X
        
    # # vecchia versione inefficiente    
    # def link(clustering, method):
        # merge_history = []
        # while len(clustering) > n_clust:
            # # ad ogni possibile coppia di label associamo la matrice delle distance
            # proximity_dict = {}
            # for lab1, lab2 in combinations(clustering.keys(), 2): 
                # dist_mtx = cdist(clustering[lab1], clustering[lab2])
                # proximity_dict[(lab1, lab2)] = dist_mtx
            # # mergiamo i cluster in base alla proximity function scelta
            # cluster_id_pair = None
            # minimum = np.inf        
            # for lab_pair, dist_mtx in proximity_dict.items(): 
                # prox_value = proximity(dist_mtx, method)
                # if prox_value < minimum:
                    # cluster_id_pair = lab_pair
                    # minimum = prox_value
            # arr = np.vstack((clustering[cluster_id_pair[0]], clustering[cluster_id_pair[1]])) # merge
            # clustering[cluster_id_pair[0]] = arr # sovrascriviamo il vecchio cluster del primo label (si può fare anche il contrio)
            # merge_history.append(cluster_id_pair)
            # del clustering[cluster_id_pair[1]] # dopo il merge cancelliamo la vecchia coppia (label, cluster) del secondo label
        # clustering = {i:cluster for i,cluster in enumerate(clustering.values())} # resettiamo i label
        # # ricomponiamo il dataset (l'ordine degli oggetti è differente) e associamo ad ogni record il suo label
        # new_labels, new_X = recompone(clustering)
        # return clustering, new_labels, new_X, merge_history
        
    def link(clustering, method):
        clustering_length = len(clustering)
        merge_history = []
        # ad ogni possibile coppia di cluster associamo la matrice delle distance
        proximity_dict = {}
        proximity_combinations = {lab:set() for lab in clustering.keys()}
        for lab1, lab2 in combinations(clustering.keys(), 2): 
            dist_mtx = cdist(clustering[lab1], clustering[lab2])
            pair = (lab1, lab2)
            proximity_dict[pair] = dist_mtx
            proximity_combinations[lab1].add(pair)
            proximity_combinations[lab2].add(pair)
        # mergiamo i 2 cluster più vicini, la vicinanza dipende dal link method scelto        
        while clustering_length > n_clust:
            cluster_id_pair = None
            minimum = np.inf        
            for lab_pair, dist_mtx in proximity_dict.items(): 
                prox_value = proximity(dist_mtx, method)
                if prox_value < minimum:
                    cluster_id_pair = lab_pair
                    minimum = prox_value
            new_cluster = np.vstack((clustering[cluster_id_pair[0]], clustering[cluster_id_pair[1]])) # merge
            # aggiorniamo la matrice delle distanze
            lab1, lab2 = cluster_id_pair
            del proximity_dict[cluster_id_pair]
            proximity_combinations[lab1].remove(cluster_id_pair)
            proximity_combinations[lab2].remove(cluster_id_pair)           
            new_comb_set = set()
            for pair1 in proximity_combinations[lab1]:
                lab3 = pair1[0]
                lab3_pos_pair1 = 0
                # se lab1 non è una tupla allora è un numpy.int32. Quando un test compara una tupla con numpy.int32, genera un ValueError
                lab1 = lab1 if isinstance(lab1, tuple) else int(lab1) 
                if pair1[0] == lab1:
                    lab3 = pair1[1]
                    lab3_pos_pair1 = 1
                pair2 = (lab2, lab3)
                lab3_pos_pair2 = 1
                if (lab2, lab3) not in proximity_combinations[lab2]:
                    pair2 = (lab3, lab2)
                    lab3_pos_pair2 = 0
                dist_mtx1 = proximity_dict[pair1]
                if lab3_pos_pair1 == 0:
                    dist_mtx1 = dist_mtx1.T
                dist_mtx2 = proximity_dict[pair2]
                if lab3_pos_pair2 == 0:
                    dist_mtx2 = dist_mtx2.T
                new_dist_mtx = np.vstack((dist_mtx1, dist_mtx2))
                del proximity_dict[pair1]
                del proximity_dict[pair2]
                proximity_dict[((lab1, lab2), lab3)] = new_dist_mtx
                new_comb_set.add(((lab1, lab2), lab3))
                proximity_combinations[lab3].remove(pair1)
                proximity_combinations[lab3].remove(pair2)
                proximity_combinations[lab3].add(((lab1, lab2), lab3))
            del proximity_combinations[lab1]
            del proximity_combinations[lab2]
            proximity_combinations[cluster_id_pair] = new_comb_set  
            # aggiorniamo il clustering        
            clustering[cluster_id_pair] = new_cluster
            merge_history.append(cluster_id_pair)
            del clustering[cluster_id_pair[0]]
            del clustering[cluster_id_pair[1]]
            clustering_length -= 1
        # resettiamo i label del clustering
        clustering = {i:cluster for i,cluster in enumerate(clustering.values())}
        # ricomponiamo il dataset (l'ordine degli oggetti è differente da quello originario) associando ad ogni record il suo label
        new_labels, new_X = recompone(clustering)
        return clustering, new_labels, new_X, merge_history        
    
    def ward_link(clustering):
        merge_history = []
        # associamo ad ogni cluster il relativo intracluster SSE
        clustering_length = len(clustering)
        sse_dict = {lab:SSE(X) for lab,X in clustering.items()}
        # cerchiamo la coppia di cluster che, una volta fusi, produce il minor incremento della total SSE
        while clustering_length > n_clust:     
            clust_id_pair = None
            sses = {}
            tot_sse_before_marging = sum(sse_dict.values())
            tot_sse_after_marging = {}
            for lab1, lab2 in combinations(clustering.keys(), 2):
                new_cluster = np.vstack((clustering[lab1], clustering[lab2]))
                new_cluster_sse = SSE(new_cluster)
                sses[(lab1, lab2)] = new_cluster_sse
                tot_sse_after_marging[(lab1, lab2)] = new_cluster_sse + tot_sse_before_marging - sse_dict[lab1] - sse_dict[lab2]
            minimum = min(tot_sse_after_marging.items(), key = lambda tpl: tpl[1])
            cluster_id_pair = minimum[0]    
            # una volta trovata la fusione più economica in termini di SSE la rendiamo effettiva
            new_cluster = np.vstack((clustering[cluster_id_pair[0]], clustering[cluster_id_pair[1]])) # merge
            clustering[cluster_id_pair] = new_cluster
            sse_dict[cluster_id_pair] = sses[cluster_id_pair]
            merge_history.append(cluster_id_pair)
            for i in range(2):
                del clustering[cluster_id_pair[i]]           
                del sse_dict[cluster_id_pair[i]]
            clustering_length -= 1        
        # resettiamo i label
        clustering = {i:cluster for i,cluster in enumerate(clustering.values())} 
        # ricomponiamo il dataset (l'ordine degli oggetti è differente) e associamo ad ogni record il suo label
        new_labels, new_X = recompone(clustering)
        return clustering, new_labels, new_X, merge_history

    # vecchia versione, non efficiente
    # def ward_link(clustering):    
        # merge_history = []
        # clustering_length = len(clustering)
        # while clustering_length > n_clust:
            # # ad ogni possibile coppia di label associamo il valore di total SSE che si otterrebbe se si fondessero proprio quei due cluster
            # sse_dict = {}
            # clust_id_pair = None
            # for lab1, lab2 in combinations(clustering.keys(), 2): 
                # aggl_clustering = deepcopy(clustering)
                # arr = np.vstack((aggl_clustering[lab1], aggl_clustering[lab2])) # merge
                # aggl_clustering[lab1] = arr
                # del aggl_clustering[lab2]
                # sse = 0
                # for X in aggl_clustering.values():
                    # sse += SSE(X)
                # sse_dict[(lab1, lab2)] = sse
            # minimum = min(sse_dict.items(), key = lambda tpl: tpl[1])
            # cluster_id_pair = minimum[0]
            # # una volta trovata la fusione più economica in termini di SSE la rendiamo effettiva
            # arr = np.vstack((clustering[cluster_id_pair[0]], clustering[cluster_id_pair[1]])) # merge
            # clustering[cluster_id_pair[0]] = arr
            # merge_history.append(cluster_id_pair)
            # del clustering[cluster_id_pair[1]]
            # clustering_length -= 1
        # # resettiamo i label
        # clustering = {i:cluster for i,cluster in enumerate(clustering.values())} 
        # # ricomponiamo il dataset (l'ordine degli oggetti è differente) e associamo ad ogni record il suo label
        # new_labels, new_X = recompone(clustering)
        # return clustering, new_labels, new_X, merge_history
    
    methods = ['single', 'complete', 'average', 'ward']
    if method not in methods:
        raise KeyError(f"the only supported methods are: {methods}")
    # associamo ad ogni label il suo cluster 
    clustering = {lab:X[labels == lab,:] for lab in np.unique(labels)}
    if method == 'ward':
        return ward_link(clustering)
    return link(clustering, method)

#Bisecting KMneans
def BKMeans(X:'dataset ndarray', n_clusters:'' = 8, init: '' = 'k-means++', n_init:'' = 10, max_iter:'' = 300, tol:'' = 1e-4, random_state:'' = None)->'-total_sse, dataset, np.array(labels)':
    data = X
    sse = 0
    iteration = 1
    clusters_dict = {}
    k = 1 # n. di clusters
    # applichiamo ricorsivamente KMeans con k = 2. Ad ogni iterazione viene partizionato in 2 il
    # cluster con il più alto SSE. se due cluster hanno lo stesso sse si scompone quello più grande.
    maxheap = [(-sse, -data.shape[0], data)] 
    hp.heapify(maxheap)
    while k != n_clusters:
        *_, data = hp.heappop(maxheap)
        model = KMeans(2, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state).fit(data)
        X0 = data[model.labels_ == 0,:]
        X1 = data[model.labels_ == 1,:]
        sse0, sse1 = intracluster_SSEs(data, model)
        hp.heappush(maxheap,(-sse0, -X0.shape[0], X0))
        hp.heappush(maxheap,(-sse1, -X1.shape[0], X1))
        k += 1
    # assegniamo i label, calcoliamo il total SSE e ricomponiamo il dataset.   
    labels = []
    total_sse, _, dataset = hp.heappop(maxheap)
    labels += [0]*dataset.shape[0]
    lab = 1
    while maxheap:
        sse, _, data = hp.heappop(maxheap)
        total_sse += sse
        dataset = np.vstack((dataset, data))
        labels += [lab]*data.shape[0]
        lab += 1
    return -total_sse, dataset, np.array(labels)    

# between group sum of squares (SSB)
def SSB(X:'', model:'clustering model or array of labels'):
    mu = X.mean(axis = 0)
    ssb = 0
    try:
        labels_arr = model.labels_
        labels, counts = np.unique(labels_arr, return_counts=True)
    # se si inserisce l'array dei labels piuttosto che il modello    
    except AttributeError:
        labels_arr = model
        labels, counts = np.unique(labels_arr, return_counts=True)
    try:
        centroids = model.cluster_centers_
    # per computare i centroidi nei i modelli non prototype    
    except AttributeError: 
        centroids = []
        for lab in labels:
            clust_centroid = X[labels_arr == lab, :].mean(axis = 0)
            centroids.append(clust_centroid)
    for i, _ in enumerate(labels):
        dist = ((centroids[i] - mu)**2).sum()
        ssb += counts[i] * dist
    return ssb