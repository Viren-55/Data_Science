# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:15:44 2020

@author: kumar
"""

import pickle
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA 

# Global variables for coordinates and labels

x_coords = []
y_coords = []

with open('dbscan2000.pkl', 'rb') as f:
    data = pickle.load(f)
    c = 0
    for line in range(len(data)):
        x_coords.append(float(data[line][0]))
        y_coords.append(float(data[line][1]))
    c+=1


db_default = DBSCAN(eps = 0.12, min_samples = 3).fit(data) 
labels = db_default.labels_ 


def plot_clustered_points(labels):
# Scatter plot graph generation function that uses different colors for different clusters
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom', cmaplist, cmap.N)

    fig = plt.figure(figsize=(10,6))
    plt.scatter(x_coords, y_coords, c=labels, s=7, cmap=cmap)
    plt.savefig('clustered_points.png') # Save the plot
    print(fig)
print(plot_clustered_points(labels))