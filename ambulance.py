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

#Class for processing data with wasserstein clustering
class Cluster:
    #initialize data with the passed file
    def __init__ (self, data, n):
        self._n = n
        '''
        dataSlice = np.random.randint(low = 922539-500, high = 922539-50)
        df = pd.read_pickle("C:/Users/rjhosler/Documents/_REU/df1.pkl")
        data = np.stack((df.XCOORD, df.YCOORD), axis = 1)
        data = data [dataSlice:922539, :]
        '''
        self._data = data

        X1 = (np.percentile((data[:,0]), 25) - np.percentile((data[:,0]), 75)) * np.random.random_sample((n,1)) + np.percentile((data[:,0]), 75)
        X2 = (np.percentile((data[:,1]), 25) - np.percentile((data[:,1]), 75)) * np.random.random_sample((n,1)) + np.percentile((data[:,1]), 75)
        X = np.stack ((X1, X2), axis = 1)
        self._centers  = X.reshape ((n,2))

    #method to implement the wasserstein algorithm
    #return new centers
    def wasserstein (self):
        theta = 1
        a = 1 / len (self._centers)
        #use intensities for b
        b = self._data [:,2] / sum (self._data [:,2])
        X = self._centers
        data = self._data [:, 0:2]
        lam = 175
        max_iter1 = 0
        while (1):
            M = cdist (X, data, 'euclidean')
            K = math.e ** (-1 * lam * M)
            Kt = np.divide (K, a)
            u = np.transpose (np.ones(len(X)) / len(X))
            u = np.float64(u)
            change = 1
            max_iter2 = 0
            while (change > 0.0001):
                oldu = u
                u = np.float64(np.transpose (np.ones(len(X)) / np.matmul (Kt, np.divide (b, np.matmul (np.transpose (K), u)))))
                if (np.isnan(u).any() or np.isinf(u).any()):
                    return (X)
                change = la.norm (u - oldu)
                if (max_iter2 > 100):
                    change = 0
                max_iter2 += 1
            V = np.divide (b, np.matmul (np.transpose(K), u))        
            T = np.matmul (np.matmul (np.diag(u), K), np.diag (V))
            oldX = X
            X = np.float64((X * (1 - theta)) + (np.divide (np.matmul (T, data), a) * theta))
            max_iter1 += 1
            if (la.norm (oldX - X) < 0.0001 or max_iter1 > 100):
                return X

    #Alternate clustering method using k-means
    def kmeans_cluster (self, init):
        if (init == True):
            kmeans = KMeans(n_clusters = len(self._centers), init = self._centers, n_init = 1).fit(self._data[:,0:2])
            self._centers = kmeans.cluster_centers_
            self._data [:, 3] = kmeans.labels_
        else:
            kmeans = KMeans(n_clusters = self._n, init = 'random').fit(self._data[:,0:2])
            self._centers = kmeans.cluster_centers_
            self._data [:, 3] = kmeans.labels_

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
        if (len(self._data [0]) == 3):
            return np.hstack ((self._data, cluster_id))
        else:
            self._data [:, 3] = np.transpose (cluster_id)
            return self._data

    #method to calculate average within cluster distance weighted by point intensities and balanced representation of clusters
    #NOTE: only works when cluster IDs have been assigned
    #return overall average
    def calc_avg_dist (self):
        avg_dist = np.empty ([0])
        for i in range (len(self._centers)):
            isEmpty = True
            size = 0
            intensity = 0
            dist_array = np.empty ([0])
            for j in range (len(self._data)):
                if (self._data [j, 3] == i):
                    dist_array = np.append (dist_array, haversine (self._centers [i, :], self._data [j, 0:2], miles = True))
                    isEmpty = False
                    size += 1
                    intensity += self._data [j, 2]
            if (isEmpty == False):
                weight = (intensity / sum (self._data [:,2])) * (size / len (self._data))
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
                if (self._data [j, 3] == i):
                    clusterSize += 1
            avg_cluster = np.append (avg_cluster, clusterSize)
        return avg_cluster, avg_cluster.var()

    #define n centers for initialization
    def set_centers (self, centers, n):
        busy = np.random.choice(np.arange(len(centers)), replace = False, size = n)
        self._centers = centers [busy]

    #randomize centers
    def randomize_centers (self):
        data = self._data
        n = self._n
        X1 = (np.percentile((data[:,0]), 25) - np.percentile((data[:,0]), 75)) * np.random.random_sample((n,1)) + np.percentile((data[:,0]), 75)
        X2 = (np.percentile((data[:,1]), 25) - np.percentile((data[:,1]), 75)) * np.random.random_sample((n,1)) + np.percentile((data[:,1]), 75)
        X = np.stack ((X1, X2), axis = 1)
        self._centers  = X.reshape ((n,2))

    #driver method, retreive best result after n iterations
    def process_data (self, n_iter):
        minDist = 100
        centers = self._centers
        data = self._data
        for i in range (n_iter):
            self._centers = self.wasserstein()
            self._data = self.cluster_assignment()
            dist = self.calc_avg_dist()
            print (dist)
            if (i == 0 or dist < minDist):
                minDist = dist
                centers = self._centers
            self.randomize_centers()

        self._centers = centers
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

class PointProcess:
    #initialize with pre-trained parameters
    def __init__ (self):
        self._Lam = np.load('Lam.npy')                  # size len(time memory), X_GRID_SIZE, YGRIDSIZE  
        self._F = np.load('F.npy')                      # size X_GRID_SIZE, YGRIDSIZE, len(w) 
        self._w = [.5, .1, .05]                   
        self._theta = np.load('theta.npy')              # size len(w)               
        self._mu = np.load('mu.npy')                    # size X_GRID_SIZE, YGRIDSIZE 
        self._last_times = np.load('last_times.npy')    # size len(time memory)        
        self._G_times = pd.read_pickle('G_times.pkl')   # size X_GRID_SIZE, YGRIDSIZE

        if self._F.shape[0:2] == self._Lam.shape[1:3] == self._G_times.shape == self._mu.shape:
            self._X_GRID_SIZE, self._Y_GRID_SIZE = self._F.shape[0:2]
        else:
            warnings.warn('Dimensions of point process initialization mismatched')
            x = [self._Lam.shape[1:3][0], self._F.shape[0:2][0], self._G_times.shape[0], self._mu.shape[0]]
            self._X_GRID_SIZE = min(x)
            y = [self._Lam.shape[1:3][1], self._F.shape[0:2][1], self._G_times.shape[1], self._mu.shape[1]]
            self._Y_GRID_SIZE = min(y)

        self._K = len(self._theta)
            
    def coord_to_grid(self, xcoord, ycoord):
    # Assign a lat and long coordinate to a grid location. Input xcoord, ycoord as lat/long to be mapped

    # Extreme values taken as max and min from 10-year EMS data
        ymin = 39.587905
        ymax = 40.0099

        xmin = -86.4619147125
        xmax =  -85.60543100000002

        if xcoord < xmin:
            xcoord = xmin
        elif xcoord > xmax:
            xcoord = xmax

        if ycoord < ymin:
            ycoord = ymin
        elif ycoord > ymax:
            ycoord = ymax  

        xbin = int((xcoord-xmin)/(xmax-xmin)*(self._X_GRID_SIZE-1))
        ybin = int((ycoord-ymin)/(ymax-ymin)*(self._Y_GRID_SIZE-1))
        return xbin, ybin

    def grid_to_coord(self, xbin, ybin):

        # Extreme values taken as max and min from 10-year EMS data
        ymin = 39.587905
        ymax = 40.0099

        xmin = -86.4619147125
        xmax =  -85.60543100000002

        xcoord = (xbin * (xmax - xmin)) / (self._X_GRID_SIZE -1) + xmin
        ycoord = (ybin * (ymax - ymin)) / (self._Y_GRID_SIZE -1) + ymin
        return xcoord, ycoord

    def update(self, event_time, xcoord, ycoord, catagory = None):    # datetime, xcoord, ycoord
        # TODO track different parameters for different catagories

        # place new event in correct grid
        gx, gy = self.coord_to_grid(xcoord, ycoord)    

        # find global time delta and append event time to event time list
        last_global_time = self._last_times[-1]
        time_delta = (event_time - last_global_time).total_seconds()
        self._last_times = np.append(self._last_times, event_time)

        # append new matrix to lambda to fill up during this update
        Lam = np.concatenate((Lam, np.zeros([1, self._X_GRID_SIZE, self._Y_GRID_SIZE])), axis=0)

        # global update of all grids
        for x in self._X_GRID_SIZE:
            for y in self._Y_GRID_SIZE:
                for k in self._K:
                    self._F[x][y][k] = self._F[x][y][k] * np.exp(-1*self._w[k]*time_delta)
                    # to prevent underflow issues:
                    if self._F[x][y][k] < 1e-70:
                        self._F[x][y][k] = 1e-70
                self._Lam[-1][x][y] = self._mu[x][y] + sum(self._F[x][y])

        # local update based on where event occurred
        dt = 0.005
        last_g_time = pd.to_datetime(G_times.at[gx,gy])
        g_time_delta = (event_time - last_g_time).total_seconds()
        self._G_times.at[gx,gy] = event_time

        if self._Lam[-1][gx][gy] == 0:
            self._Lam[-1][gx][gy] = 1e-70
        self._mu[gx][gy] = self._mu[gx][gy] + dt * (self._mu[gx][gy]/self._Lam[-1][gx][gy] * g_time_delta)
        for k in self._K:
            self._theta[k] = self._theta[k] + dt * (self._F[gx][gy][k]/self._Lam[-1][gx][gy] - self._theta[k])
            self._F[gx][gy][k] = self._F[gx][gy][k] + self._w[k]*self._theta[k]

        #reindex last times and Lambda
        time_memory = 30    # Number of past event lambads and times to save
        if time_memory <= min(len(self._Lam), len(self._last_times)):
            self._Lam = self._Lam[-time_memory:]
            self._last_times = self._last_times[-time_memory:,:]

        # could also choose to pickle everything here
        #self._G_times.to_pickle('G_times.pkl')
        #np.save('last_times.npy', self._last_times)
        #np.save('Lam.pkl', self._Lam)
        #np.save('F.pkl', self._F)
        #np.save('theta.pkl', self._theta)
        #np.save('mu.pkl', self._mu)

    def intensity_snapshot(self, catagory = None):
        # returns an array of coordinates and their intensity for front end (yes, currently the same as locs_for_wasserstein...)
        x_y_lam = np.empty((0,0,0))

        for x in range(0, self._X_GRID_SIZE):
            for y in range(0, self._Y_GRID_SIZE):
                xcoord, ycoord = self.grid_to_coord(x, y)
                # for now just sending over lambda snapshot @ current prediction...
                lam = self._Lam[-1][x][y]
                to_append = xcoord, ycoord, lam
                np.append(x_y_lam, to_append)
        
        x_y_lam = x_y_lam.reshape ((len(x_y_lam)//3,3))
        return x_y_lam

    def sms_project(self, catagory = None):
        # returns flag to send SMS if projected events exceed certain level
        # for now randomly return True or False to simulate flag to send sms
        # TODO
        probability = .3
        return random.random() < probability

    def locs_for_wasserstein(self, catagory = None):
        # returns an array of points for Waserstein [[x, y, lambda], [...]]
        x_y_lam = np.empty((0,0,0))

        for x in range(0, self._X_GRID_SIZE):
            for y in range(0, self._Y_GRID_SIZE):
                xcoord, ycoord = self.grid_to_coord(x, y)
                # for now just sending over lambda snapshot @ current prediction...
                lam = self._Lam[-1][x][y]
                to_append = xcoord, ycoord, lam
                x_y_lam = np.append(x_y_lam, to_append)

        x_y_lam = x_y_lam.reshape ((len(x_y_lam)//3,3))
        return x_y_lam

    
from flask import Flask, redirect, url_for, request, jsonify     
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    npzfile = np.load ('grid_loc1.npz')
    grid_loc = npzfile['arr_0']
    #grid_loc = np.delete (grid_loc, (19), axis = 0)
    '''
    plt.scatter(grid_loc [:,0], grid_loc [:,1])
    plt.title("Number of events in each grid")
    plt.show()
    '''
    grid_loc [:,2] = np.log (grid_loc [:,2])
    n = 30 #np.random.randint(low = 15, high = 30)
    cluster = Cluster (grid_loc, n)

    #init wasserstein
    cluster.process_data(10)
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()

    plt.title ('Wasserstein')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:,0], centers[:,1], c = 'red', s = 100, alpha=0.5)
    plt.show()
    
    summary1 = [dist]
    
    #init kmeans
    cluster.process_data_kmeans(False)
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()

    plt.title ('Kmeans')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:,0], centers[:,1], c = 'red', s = 100, alpha=0.5)
    plt.show()
    
    summary2 = [dist]
    
    return jsonify ([summary1, summary2])

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