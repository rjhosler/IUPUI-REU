#import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import io
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
from PointProcess import PointProcessTrain
import time

#Class for processing data with wasserstein clustering
class Cluster:
    #initialize data with the passed file
    def __init__ (self, data, n):
        self._n = n
        self._data = data

        points = np.random.choice(np.arange(len(data)), replace = False, size = n)
        self._centers = data [points, 0:2]
        self._oldcenters = self._centers

    #method to implement the wasserstein algorithm
    #return new centers
    def wasserstein (self, lam):
        theta = 0.5
        a = 1 / len (self._centers)
        #use intensities for b
        b = self._data [:,2] / sum (self._data [:,2])
        X = self._centers
        data = self._data [:, 0:2]
        max_iter1 = 0
        while (1):
            #usable distance metrics: chebyshev, cityblock, euclidean 
            M = cdist (X, data, 'euclidean')
            K = math.e ** (-1 * lam * M)
            Kt = np.divide (K, a)
            u = np.transpose (np.ones(len(X)) / len(X))
            u = np.float64(u)
            change = 1
            max_iter2 = 0
            while (change > 0.001):
                oldu = u
                u = np.float64(np.transpose (np.ones(len(X)) / np.matmul (Kt, np.divide (b, np.matmul (np.transpose (K), u)))))
                if (np.isnan(u).any() or np.isinf(u).any()):
                    print ("emergency exit")
                    return (X)
                change = la.norm (u - oldu)
                if (max_iter2 > np.amax([lam*2, 100])):
                    change = 0
                max_iter2 += 1
            V = np.divide (b, np.matmul (np.transpose(K), u))        
            T = np.matmul (np.matmul (np.diag(u), K), np.diag (V))
            oldX = X
            X = np.float64((X * (1 - theta)) + (np.divide (np.matmul (T, data), a) * theta))
            max_iter1 += 1
            if (la.norm (oldX - X) < 0.001 or max_iter1 > 100):
                return X

    #Alternate clustering method using k-means
    def kmeans_cluster (self, init):
        if (init == True):
            kmeans = KMeans(n_clusters = len(self._oldcenters), init = self._oldcenters, n_init = 1, max_iter = 10).fit(self._data[:,0:2])
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
                dist = la.norm (self._data [i, 0:2] - self._centers [j, :])
                if (j == 0 or dist < minDist):
                    minDist = dist
                    cluster_id [i] = j
        if (len(self._data [0]) == 3):
            return np.hstack ((self._data, cluster_id))
        else:
            self._data [:, 3] = np.transpose (cluster_id)
            return self._data

    #remove points that are closest to trucks that cannot move
    def remove_points (self, still_data):
        before = len(self._data)
        iter_balance = 0
        for i in range (len(self._data)):
            i = i - iter_balance
            for j in range (len(self._centers)):
                dist = la.norm (self._data [i, 0:2] - self._centers [j, :])
                if (j == 0 or dist < minDist):
                    minDist = dist
            #test for points that cannot move. if a point is closer to any of those points, remove it from the data
            test_condition = True
            rm_iter = 0
            while (test_condition):
                dist = la.norm (self._data [i, 0:2] - still_data [rm_iter, :])
                if (dist < minDist):
                    self._data = np.delete (self._data, i, 0)
                    iter_balance += 1
                    test_condition = False
                rm_iter += 1
                if (rm_iter == len (still_data)):
                    test_condition = False
        print (before - len(self._data), "points removed")                 
            

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
                    #dist_array = np.append (dist_array, self.driving_distance (self._centers [i, :], self._data [j, 0:2]))
                    isEmpty = False
                    size += 1
                    intensity += self._data [j, 2]
            if (isEmpty == False):
                weight = intensity / sum (self._data [:,2])
                avg_dist = np.append (avg_dist, dist_array.mean() * weight)
        return sum(avg_dist)

    #Method to centralize points based on cluster's coordinates
    #NOTE: requires cluster assignment
    def round_off (self):
        for i in range (len(self._centers)):
            point = [[xcoord,ycoord] for (xcoord,ycoord,intensity,cluster) in self._data if cluster == i]
            point = np.array (point)
            if (point.size != 0):
                self._centers [i] = point.mean(axis = 0)
        self.cluster_assignment()

    #define n centers for initialization
    def set_centers (self, centers, n):
        busy = np.random.choice(np.arange(len(centers)), replace = False, size = n)
        self._centers = centers [busy]
        self._oldcenters = self._centers

    #randomize centers
    def randomize_centers (self):
        data = self._data
        n = self._n
        points = np.random.choice(np.arange(len(data)), replace = False, size = n)
        self._centers = data [points, 0:2]
        

    #method for learning smoothing parameter
    def learn_lam (self, n_iter, rand_centers, lam):
        centers = self._centers
        data = self._data
        #lam = np.random.randint(low = len(self._data) / 2, high = len(self._data))
        minDist = 100; prev_lam = 0; flam = 0; dist = 10; low = 1; high = 5; diminish = 1; found = 0
        for i in range (n_iter):
            self._centers = self.wasserstein(lam)
            self._data = self.cluster_assignment()
            dist = self.calc_avg_dist()
            stats = [dist, lam]
            print (stats)
            if (i == 0 or dist < minDist):
                found = i
                minDist = dist
                centers = self._centers
                if (diminish < 0.1):
                    diminish = 0.2
                if (prev_lam <= lam):
                    flam = lam
                    lam += high * 4 * diminish
                    diminish += -0.05
                elif (prev_lam > lam):
                    flam = lam
                    lam -= high * 1 * diminish
                    diminish += -0.05
            elif (prev_dist <= dist and prev_lam <= lam):
                prev_lam = lam
                lam -= np.random.randint(low = low, high = high)
            elif (prev_dist > dist and prev_lam > lam):
                prev_lam = lam
                lam -= np.random.randint(low = low, high = high)
            elif (prev_dist <= dist and prev_lam > lam):
                prev_lam = lam
                lam += np.random.randint(low = low, high = high)
            elif (prev_dist > dist and prev_lam <= lam):
                prev_lam = lam
                lam += np.random.randint(low = low, high = high)
            else:
                prev_lam = lam
                lam += np.random.randint(low = low, high = high)
            prev_dist = dist
            if (rand_centers == False and i % 2 == 0):
                self._centers = self._oldcenters
            else:
                self.randomize_centers()

        self._centers = centers
        self._data = self.cluster_assignment()
        #print ("Iteration: ", found + 1)
        return flam

    #calc driving distance between 2 coordinates
    def driving_distance (self, coord1, coord2):
        orig_coord = "{0},{1}".format(str(coord1 [0]),str(coord1 [1]))
        dest_coord = "{0},{1}".format(str(coord2 [0]),str(coord2 [1]))
        url = "http://maps.googleapis.com/maps/api/distancematrix/json?origins={0}&destinations={1}&mode=driving&language=en-EN&sensor=false".format(str(orig_coord),str(dest_coord))
        result = simplejson.load(urllib.request.urlopen(url))
        if (result['status'] == 'OVER_QUERY_LIMIT'):
            print ('oof. try again')
            print (result)
            time.sleep(1)
            self.driving_distance(coord1, coord2)            
        driving_time = result['rows'][0]['elements'][0]['distance']['value'] / 1000
        return driving_time

    #driver method for kmeans
    def process_data_kmeans (self, init):
        self._data = self.cluster_assignment()
        if (init == True):
            self.kmeans_cluster (True)
        else:
            self.kmeans_cluster (False)

    #return average distance
    def get_dist (self):
        return self.calc_avg_dist()

    #return data
    def get_data (self):
        return self._data

    #return centers
    def get_centers (self):
        return self._centers
