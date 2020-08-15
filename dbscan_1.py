'''
    DBscan algorithm as per the assginment in BT3041

'''

import pickle, random
import numpy as np
import matplotlib.pyplot as plt

def Dbscan(D, eps, minpts): 
    '''
        D = data set
        eps = epsilon radius
        minpts = minpts in an area
    '''
    labels = [0]*D.shape[0]

    C = 0

    # check for neighbouring pts and grow cluster or identify noise point
    for p in range(len(D)):
        if labels[p] == 0:
            Neighbour_pts = check(D,p,eps)

            if len(Neighbour_pts) < minpts:
                labels[p] = -1 # identify noise point
            
            else:
                C += 1
                labels = update_cluster(D, labels, p, Neighbour_pts, C, eps, minpts)

    return labels

def check(D,p,eps):
    # check for the neighbours of a given p point and return the index of the point in the actual data
    neighbors_ind = []

    for k in range(len(D)):
        if np.linalg.norm(D[p]-D[k]) < eps :
            neighbors_ind.append(k)

    return neighbors_ind


def update_cluster(D, labels, p, Neighbour_pts, C, eps, minpts):
    # grow the cluster using dbscan algorithm
    labels[p] = C

    i = 0
    while i < len(Neighbour_pts):
        k = Neighbour_pts[i]

        if labels[k] == -1:
            labels[k] = C

        elif labels[k]==0:
            labels[k] = C

            k_neighbors = check(D,k,eps)
            
            if len(k_neighbors) >= minpts:
                Neighbour_pts = Neighbour_pts + k_neighbors
        
        i+=1
    return labels

# upload data
with open("dbscan2000.pkl", "rb") as f:
    d = pickle.load(f)
    f.close()

clusters = Dbscan(d, 0.2, 20) 
# print(clusters)
seg_pts = [] # list of all the segregated points

# training and plotting
for j in range(1, max(clusters)+1):
    cluster_pts = []
    rgb = (random.random(), random.random(), random.random())
    for jj in range(len(clusters)):
        if clusters[jj] == j:
            cluster_pts.append(d[jj,:])
            plt.scatter(d[jj,0], d[jj,1], c=[rgb])
    seg_pts.append(cluster_pts)

print(len(seg_pts))
plt.show()
# plt.scatter()
