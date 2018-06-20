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
import datetime as datetime
import random
import warnings


#Class for processing data with wasserstein clustering
class Cluster:
    #initialize data with the passed file
    def __init__ (self, filename):
        df = pd.read_pickle("C:/Users/rjhosler/Documents/_REU/df.pkl")
        data = np.stack((df.XCOORD, df.YCOORD), axis = 1)
        data = data [922439:922539, :]
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
        lam = 300
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
            if (la.norm (oldX - X) < 0.0025 or max_iter1 > 50):
                return X

    #Alternate clustering method using k-means
    def kmeans_cluster (self):
        kmeans = KMeans(n_clusters = 30).fit(self._data[:,0:2])
        self._centers = kmeans.cluster_centers_
        self._data [:, 2] = kmeans.labels_

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

    #driver method for kmeans
    def process_data_kmeans (self):
        self.kmeans_cluster()

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
        self._last_times = pd.load('last_times.npy')    # size len(time memory)        
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
        Lam = np.concatenate((Lam, np.zeros([1, X_GRID_SIZE, Y_GRID_SIZE])), axis=0)

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
            self._last_times = self._last_times[[-time_memory:,:]]

        # could also choose to pickle everything here
        #self._G_times.to_pickle('G_times.pkl')
        #self._last_times.to_pickle('last_times.pkl')
        #np.save('Lam.pkl', self._Lam)
        #np.save('F.pkl', self._F)
        #np.save('theta.pkl', self._theta)
        #np.save('mu.pkl', self._mu)

    def intensity_snapshot(self, catagory = None):
        # returns an array of coordinates and their intensity for front end (yes, currently the same as locs_for_wasserstein...)
        x_y_lam = np.empty((0,0,0))

        for x in self._X_GRID_SIZE:
            for y in self._Y_GRID_SIZE:
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

        for x in self._X_GRID_SIZE:
            for y in self._Y_GRID_SIZE:
                xcoord, ycoord = self.grid_to_coord(x, y)
                # for now just sending over lambda snapshot @ current prediction...
                lam = self._Lam[-1][x][y]
                to_append = xcoord, ycoord, lam
                np.append(x_y_lam, to_append)

        x_y_lam = x_y_lam.reshape ((len(x_y_lam)//3,3))
        return x_y_lam


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

    summary1 = [dist, clusterVar]

    data = data.tolist()
    centers = centers.tolist()
    clusterSizes1 = clusterSizes.tolist()

    cluster.process_data_kmeans()
    data = cluster.get_data()
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    clusterSizes, clusterVar = cluster.get_cluster_stats()

    plt.title ('kmeans')
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)
    plt.show()

    summary2 = [dist, clusterVar]

    data = data.tolist()
    centers = centers.tolist()
    clusterSizes2 = clusterSizes.tolist()

    summary = [clusterSizes1, summary1, clusterSizes2, summary2]

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
