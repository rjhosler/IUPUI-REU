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

#Class for processing data with wasserstein clustering
class Cluster:
    #initialize data with the passed file
    def __init__ (self, filename):
        df = pd.read_pickle("C:/Users/rjhosler/Documents/_REU/df.pkl")
        data = np.stack((df.XCOORD, df.YCOORD), axis = 1)
        data = data [0:1000, :]
        self._data = data

        X1 = (np.amin((data[:,0])) - np.amax((data[:,0]))) * np.random.random_sample((30,1)) + np.amax((data[:,0]))
        X2 = (np.amin((data[:,1])) - np.amax((data[:,1]))) * np.random.random_sample((30,1)) + np.amax((data[:,1]))
        X = np.stack ((X1, X2), axis = 1)
        self._centers  = X.reshape ((30,2))       

    #method to implement the wasserstein algorithm
    #return new centers
    def wasserstein (self):
        theta = 0.5
        a = len (self._data)
        b = len (self._centers)
        X = self._centers
        data = self._data
        lam = 150
        max_iter = 0
        while (1):
            M = cdist (X, data, 'euclidean')
            K = math.e ** (-1 * lam * M)
            Kt = np.divide (K, a)
            u = np.transpose (np.ones(len(X)) / len(X))
            change = 1
            while (change > 0.0001):
                oldu = u
                u = np.transpose (np.ones(len(X)) / np.matmul (Kt, np.divide (b, np.matmul (np.transpose (K), u))))
                change = la.norm (u - oldu)
            V = np.divide (b, np.matmul (np.transpose(K), u))        
            T = np.matmul (np.matmul (np.diag(u), K), np.diag (V))
            oldX = X
            X = (X * (1 - theta)) + (np.divide (np.matmul (T, data), a) * theta)
            print (la.norm (oldX - X))
            max_iter += 1
            if (la.norm (oldX - X) < 0.001 or max_iter > 50):
                return X

    #Assign cluster ID's by proximity
    #return data concatentated with cluster_id
    def cluster_assignment (self):
        cluster_id = np.zeros ((len(self._data), 1))
        for i in range (len(self._data)):
            for j in range (len(self._centers)):
                dist = la.norm (self._data [i,:] - self._centers [j,:])
                if (j == 0 or dist < minDist):
                    minDist = dist
                    cluster_id [i] = j
        return np.hstack ((self._data, cluster_id))

    #method to calculate average within cluster distance
    #NOTE: only works when cluster IDs have been assigned
    #return overall average
    def calc_avg_dist (self):
        avg_dist = np.empty ([0])
        for i in range (len(self._centers)):
            isEmpty = True
            dist_array = np.empty ([0])
            for j in range (len(self._data)):
                if (self._data [j, 2] == i):
                    dist_array = np.append (dist_array, haversine (self._centers [i, :], self._data [j, 0:2], miles = True))
                    isEmpty = False
            if (isEmpty == False):
                avg_dist = np.append (avg_dist, dist_array.mean())
        return avg_dist.mean()

    #method to return statistics on cluster size
    #NOTE: this method only works when cluster IDs have been assigned
    #return cluster sizes and cluster size variance
    def cluster_size_stats (self):
        avg_cluster = np.empty ([0])
        for i in range (len(self._centers)):
            clusterSize = 0
            for j in range (len(self._data)):
                if (self._data [j, 2] == i):
                    clusterSize += 1
            avg_cluster = np.append (avg_cluster, clusterSize)
        return avg_cluster, avg_cluster.var()
        

    #driver method
    def process_data (self):
        self._centers = self.wasserstein()
        self._data = self.cluster_assignment()

    #return average distance
    def get_dist (self):
        return self.calc_avg_dist()

    #return cluster size statistics
    def get_cluster_stats (self):
        return self.cluster_size_stats()

    #return data
    def get_data (self):
        return self._data

    #return centers
    def get_centers (self):
        return self._centers

from flask import Flask, redirect, url_for, request, jsonify     
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    cluster = Cluster (name)
    cluster.process_data()
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()

    plt.title ('Wasserstein')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)
    plt.show()

    summary = [dist, clusterVar]

    data = data.tolist()
    centers = centers.tolist()
    clusterSizes = clusterSizes.tolist()

    return jsonify (summary)

@app.route('/login', methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form.get('nm', None)
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

if __name__ == '__main__':
   app.run()
