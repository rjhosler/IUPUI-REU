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
    def __init__ (self, filename, n):
        self._n = n
        dataSlice = np.random.randint(low = 922539-500, high = 922539-50)
        df = pd.read_pickle("C:/Users/rjhosler/Documents/_REU/df1.pkl")
        data = np.stack((df.XCOORD, df.YCOORD), axis = 1)
        data = data [dataSlice:922539, :]
        self._data = data

        X1 = (np.amin((data[:,0])) - np.amax((data[:,0]))) * np.random.random_sample((n,1)) + np.amax((data[:,0]))
        X2 = (np.amin((data[:,1])) - np.amax((data[:,1]))) * np.random.random_sample((n,1)) + np.amax((data[:,1]))
        X = np.stack ((X1, X2), axis = 1)
        self._centers  = X.reshape ((n,2))

    #method to implement the wasserstein algorithm
    #return new centers
    def wasserstein (self):
        theta = 0.5
        a = len (self._data)
        b = len (self._centers)
        X = self._centers
        data = self._data [:, 0:2]
        lam = 200
        max_iter1 = 0
        while (1):
            M = cdist (X, data, 'euclidean')
            K = math.e ** (-1 * lam * M)
            Kt = np.divide (K, a)
            u = np.transpose (np.ones(len(X)) / len(X))
            change = 1
            max_iter2 = 0
            while (change > 0.0001 or max_iter2 < 100):
                oldu = u
                u = np.transpose (np.ones(len(X)) / np.matmul (Kt, np.divide (b, np.matmul (np.transpose (K), u))))
                change = la.norm (u - oldu)
                max_iter2 += 1
            V = np.divide (b, np.matmul (np.transpose(K), u))        
            T = np.matmul (np.matmul (np.diag(u), K), np.diag (V))
            oldX = X
            X = (X * (1 - theta)) + (np.divide (np.matmul (T, data), a) * theta)
            max_iter1 += 1
            print (la.norm (oldX - X))
            if (la.norm (oldX - X) < 0.0001 or max_iter1 > 100):
                return X

    #Alternate clustering method using k-means
    def kmeans_cluster (self, init):
        if (init == True):
            kmeans = KMeans(n_clusters = len(self._centers), init = self._centers, n_init = 1).fit(self._data[:,0:2])
            self._centers = kmeans.cluster_centers_
            self._data [:, 2] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters = self._n, init = 'random').fit(self._data[:,0:2])
            self._centers = kmeans.cluster_centers_
            self._data [:, 2] = kmeans.labels_

    #Assign cluster ID's by proximity
    #return data concatentated with cluster_id
    def cluster_assignment (self):
        cluster_id = np.zeros ((len(self._data), 1))
        for i in range (len(self._data)):
            for j in range (len(self._centers)):
                dist = la.norm (self._data [i, 0:2] - self._centers [j,:])
                if (j == 0 or dist < minDist):
                    minDist = dist
                    cluster_id [i] = j
        if (len(self._data [0]) == 2):
            return np.hstack ((self._data, cluster_id))
        else:
            self._data [:, 2] = np.transpose (cluster_id)
            return self._data

    #method to calculate average within cluster distance weighted by the probability of the cluster and the size of the cluster for even representation
    #NOTE: only works when cluster IDs have been assigned
    #return overall average
    def calc_avg_dist (self):
        avg_dist = np.empty ([0])
        for i in range (len(self._centers)):
            isEmpty = True
            size = 0
            dist_array = np.empty ([0])
            for j in range (len(self._data)):
                if (self._data [j, 2] == i):
                    dist_array = np.append (dist_array, haversine (self._centers [i, :], self._data [j, 0:2], miles = True))
                    isEmpty = False
                    size += 1
            if (isEmpty == False):
                weight = (size ** 2) / len(self._data)
                avg_dist = np.append (avg_dist, dist_array.mean() * weight)
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

    #define n centers for initialization
    def set_centers (self, centers, n):
        busy = np.random.choice(np.arange(len(centers)), replace = False, size = n)
        self._centers = centers [busy]

    #driver method
    def process_data (self):
        self._centers = self.wasserstein()
        self._data = self.cluster_assignment()

    #driver method for kmeans
    def process_data_kmeans (self, init):
        if (init == True):
            self.kmeans_cluster (True)
        else:
            self.kmeans_cluster (False)

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
    n = np.random.randint(low = 10, high = 30)
    cluster = Cluster (name, n)

    #init wasserstein
    cluster.process_data()
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()
    '''
    plt.title ('Wasserstein')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)
    plt.show()
    '''
    summary1 = [dist]

    #update wasserstein
    cluster.set_centers(centers, n-1)
    cluster.process_data()
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()
    '''
    plt.title ('Wasserstein2')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)
    plt.show()
    '''
    summary2 = [dist]

    #update wasserstein again
    cluster.set_centers(centers, n-2)
    cluster.process_data()
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()
    '''
    plt.title ('Wasserstein3')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)
    plt.show()
    '''
    summary3 = [dist]

    #init kmeans
    cluster.process_data_kmeans(False)
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()
    '''
    plt.title ('kmeans')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)
    plt.show()
    '''
    summary4 = [dist]

    #update kmeans
    cluster.set_centers(centers, n-1)
    cluster.process_data_kmeans(True)
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()
    '''
    plt.title ('kmeans2')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)
    plt.show()
    '''
    summary5 = [dist]

    #update kmeans again
    cluster.set_centers(centers, n-2)
    cluster.process_data_kmeans(True)
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()
    '''
    plt.title ('kmeans3')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)
    plt.show()
    ''' 
    summary6 = [dist]

    summary = [summary1, summary2, summary3, summary4, summary5, summary6]

    return jsonify (summary)

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
