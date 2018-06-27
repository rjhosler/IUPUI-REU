import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import io
import matplotlib.pylab as pl
from scipy import linalg as la
from numpy.linalg import inv
import math
from scipy.spatial.distance import cdist
from haversine import haversine
from math import *
from random import *
import datetime as datetime
from tempfile import TemporaryFile
import datetime as datetime
import random
import warnings
from scipy import stats
import simplejson, urllib
import urllib.request
import wasserstein
import PointProcess
from wasserstein import Cluster
from PointProcess import PointProcessTrain

#process data for clustering
def create_data ():
    pdata = np.load('locs.npy')
    grid_loc = np.empty((0,0))
    pdata[:,2] = pdata[:,2] / 100
    lst = pdata[:,2].tolist()
    scom = max(set(lst), key=lst.count)
    
    for i in range (len(pdata)):
        if (pdata [i,2] != scom and pdata [i,2] > 0.0):
            loc = pdata[i]
            grid_loc = np.append(grid_loc, loc)
    grid_loc = grid_loc.reshape ((len(grid_loc) // 3, 3))
    
    #print (len(grid_loc))
    #print (len(pdata) - len(grid_loc))
    size = np.random.randint(low = 100 , high = len(grid_loc) // 2)
    points = np.random.choice(np.arange(len(grid_loc)), replace = False, size = size)
    grid_loc = grid_loc [points]
    #grid_loc = grid_loc [410:610]

    return grid_loc


from flask import Flask, redirect, url_for, request, jsonify     
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    graph_size = 100
    grid_loc = create_data()
    print ('data size: ', len(grid_loc))
    '''
    plt.scatter(grid_loc [:,0], grid_loc [:,1])
    plt.title("Number of events in each grid")
    plt.show()
    '''    
    n = np.random.randint(low = 15, high = 35)
    print ('cluster size: ', n)
    cluster = Cluster (grid_loc, n)
    #d = cluster.driving_distance ([40.730610, -73.935242], [38.889931, -77.009003])
    
    #init wasserstein
    #cluster.process_data(10)
    lam = cluster.learn_lam(20, True)
    print (lam)
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()

    plt.title ('Wasserstein')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:,0], centers[:,1], c = 'red', s = graph_size, alpha = 0.5)
    plt.show()
    
    summary1 = [dist]

    #init wasserstein + round_off combo
    #cluster.process_data(10)
    #cluster.process_data_kmeans(True)
    cluster.round_off()
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()

    plt.title ('Wasserstein + Round_Off')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:,0], centers[:,1], c = 'red', s = graph_size, alpha = 0.5)
    plt.show()
    
    summary2 = [dist]
    
    #init kmeans
    cluster.process_data_kmeans(False)
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()

    plt.title ('Kmeans')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:,0], centers[:,1], c = 'red', s = graph_size, alpha = 0.5)
    plt.show()
    
    summary3 = [dist]
    
    return jsonify ([summary1, summary2, summary3])
    
@app.route('/login', methods = ['POST', 'GET'])
def login():
   if (request.method == 'POST'):
      user = request.form.get('nm', None)
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

if __name__ == '__main__':
   app.run()
