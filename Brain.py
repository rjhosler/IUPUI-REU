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

from flask import Flask, redirect, url_for, request, jsonify     
app = Flask(__name__)

@app.route('/emergencies')
def emergencies():
    start_time = request.args.get('start_time')
    interval_count = request.args.get('interval_count')
    if (request.args.get('time_interval')):
        time_interval = request.args.get('time_interval')
        time_interval = int(time_interval)
    else:
        time_interval = 15
 
    Date =  datetime.datetime.fromtimestamp(float(start_time))
    total_output = []
    sample = np.load ('SampleOutputs.npz')
    loc = sample['intensities']
           
    for j in range (int(interval_count)):            
        output = {
            'start': Date + datetime.timedelta(0,60*time_interval*j),
            'interval_length': time_interval,
            'emergencies': []
        }
        for i in range (len(loc[0])):
            output ['emergencies'].append({
                'intensity': loc [j][i,2],
                'location': {
                    'lat': loc [j][i,0],
                    'long': loc [j][i,1]
                    }
            })
        total_output.append (output)
    return jsonify(total_output)

@app.route('/assignments', methods = ['POST'])
def assignments():
    if (request.method == 'POST'):

        sample = np.load ('SampleOutputs.npz')
        em_data = sample ['intensities']
    
        data = request.get_json()
        
        trucks = filter_data(data ['trucks'])
        start_time = datetime.datetime.fromtimestamp(float(data ['start_time']))
        interval_time = data ['interval_time']
        interval_count = data ['interval_count']
        virtual = trucks [:,2]
        
        #assignments = dummy_data (trucks, interval_time, interval_count, em_data)
        assignments = wasserstein_cluster (trucks, interval_time, interval_count, em_data)

        output = {
            'date': start_time,
            'intervals': interval_count
        }
        output ['TruckIntervalSchema'] = {
            'date': start_time,
            'trucks': len(trucks)
        }
        output ['TruckSchema'] = []
        
        assign_iter = 0
        for i in range (len(trucks)):
            curr_object = {
                'id': trucks [i,3],
                'location': {
                    'lat': trucks [i,0],
                    'long': trucks [i,1]
                }
            }
            if (trucks [i,2] == True):
                curr_object ['assigned_location'] = {
                    'lat': assignments [assign_iter,0],
                    'long': assignments [assign_iter,1]
                }
                assign_iter += 1
            else:
                curr_object ['assigned_location'] = curr_object ['location']
            output ['TruckSchema'].append (curr_object)
            
        return jsonify(output)
        
def filter_data (data):
    trucks = np.zeros((len(data),5))
    for i in range (len(data)):
        trucks [i,0] = data [i]['location']['lat']
        trucks [i,1] = data [i]['location']['long']
        trucks [i,2] = data [i]['virtual']
        trucks [i,3] = data [i]['id']
        trucks [i,4] = data [i]['type']
    return trucks
    
def dummy_data (trucks, interval_time, interval_count, em_data):    
    data = np.zeros((len(trucks),3))
    data[:,0] = trucks [:,0]
    data[:,1] = trucks [:,1]
    data[:,2] = trucks [:,2]
    data = data.tolist()
    data = [[lat, long, virtual] for (lat, long, virtual) in data if virtual == True]
    data = np.array(data)

    kmeans = KMeans(n_clusters = len(data)).fit(data[:,0:2])
    centers = kmeans.cluster_centers_

    return centers

#practical assignments
'''
    em_data is the locs for wasserstein over the given interval
    2.) Call Cluster in wasserstein passing em_data and len(all trucks with virtual = true)
    3.) Initialize centers with the locations of the true trucks
    4.) Cluster the data and return the centers
'''
def wasserstein_cluster (trucks, interval_time, interval_count, em_data):
    grid_loc, data = shrink_data (em_data, interval_count, trucks)
    data = np.zeros((len(trucks),3))

    cluster = Cluster (grid_loc, len(data))
    cluster.set_centers (data[:,0:2], len(data))
    lam = cluster.learn_lam(10, False)
    centers = cluster.get_centers()
    data = cluster.get_data()    
    return centers

def shrink_data (em_data, interval_count, trucks):
    data = np.zeros((len(trucks),3))
    data[:,0] = trucks [:,0]
    data[:,1] = trucks [:,1]
    data[:,2] = trucks [:,2]
    data = data.tolist()
    data = [[lat, long, virtual] for (lat, long, virtual) in data if virtual == True]
    data = np.array(data)

    em_shrink = sum (em_data [0:interval_count,:])
    em_shrink [:,0:2] = em_shrink [:,0:2] / interval_count
    grid_loc = np.empty((0,0))
    lst = em_shrink[:,2].tolist()
    scom = max(set(lst), key=lst.count)
    
    for i in range (len(em_shrink)):
        if (em_shrink [i,2] != scom and em_shrink [i,2] > 0.0):
            loc = em_shrink[i]
            grid_loc = np.append(grid_loc, loc)
    grid_loc = grid_loc.reshape ((len(grid_loc) // 3, 3))
    temp = np.copy(grid_loc [:,0])
    grid_loc [:,0] = grid_loc [:,1]
    grid_loc [:,1] = temp

    return grid_loc, data

if __name__ == '__main__':
   app.run()