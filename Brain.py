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
    '''
    fields = ['XCOORD', 'YCOORD', 'CALL_TYPE_FINAL_D', 'CALL_TYPE_FINAL', 'DATE_TIME']
    full_data = pd.read_csv('FixedLatLongTimeLoc_AndTimeSorted.csv', usecols=fields)
    full_data['DATE_TIME'] =  pd.to_datetime(full_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    full_data = full_data.sort_values(by='DATE_TIME')
    data = full_data[300000:305000]
    data.reset_index(drop=True, inplace=True)
    
    pObject = PointProcessTrain(data)
    pdata = pObject.test_locs_for_wasserstein()
    print (pdata)
    '''
    pdata = np.load('locs.npy')
    grid_loc = np.empty((0,0))
    lst = pdata[:,2].tolist()
    scom = max(set(lst), key=lst.count)
    
    for i in range (len(pdata)):
        if (pdata [i,2] != scom and pdata [i,2] > 0.0):
            loc = pdata[i]
            grid_loc = np.append(grid_loc, loc)
    grid_loc = grid_loc.reshape ((len(grid_loc) // 3, 3))
    
    #print (len(grid_loc))
    #print (len(pdata) - len(grid_loc))
    size = np.random.randint(low = 100 , high = 250)
    points = np.random.choice(np.arange(len(grid_loc)), replace = False, size = size)
    grid_loc = grid_loc [points]
    #grid_loc = grid_loc [410:610]

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
    
    #return jsonify ([summary1, summary2, summary3])
    
    return grid_loc


from flask import Flask, redirect, url_for, request, jsonify     
app = Flask(__name__)

@app.route('/emergencies')
def emergencies():
    start_time = request.args.get('start_time')
    interval_count = request.args.get('interval_count')
    if (request.args.get('time_interval')):
        time_interval = request.args.get('time_interval')
    else:
        time_interval = 15
 
    Date =  datetime.datetime.fromtimestamp(float(start_time))
    lonmin = -86.4619147125; lonmax =  -85.60543100000002; latmin = 39.587905; latmax = 40.0099;
    total_output = []
    
    #for now, generate 100 fake locations        
    for j in range (int(interval_count)):
        loc = np.zeros((100,3))
        for i in range (100):
            loc [i,0] = np.random.uniform(lonmin, lonmax)
            loc [i,1] = np.random.uniform(latmin, latmax)
            loc [i,2] = np.random.uniform(0, 1)
            
        output = {
            'start': Date + datetime.timedelta(0,60*time_interval*j),
            'interval_length': time_interval
        }

        for i in range (len(loc)):
            if ('emergencies' in output):
                output ['emergencies'].update ({
                    'intensity': loc[i,2],
                    'location': {
                        'lat': loc [i,0],
                        'long': loc [i,1]
                        }
                })
            else:
                output ['emergencies'] = {
                    'intensity': loc[i,2],
                    'location': {
                        'lat': loc [i,0],
                        'long': loc [i,1]
                        }
                }
        total_output.append (output)
    return jsonify(total_output)

@app.route('/assignments', methods = ['POST'])
def assignments():
    if (request.method == 'POST'):
        data = request.get_json()
        trucks = data ['trucks']
        interval_time = data ['interval_time']
        interval_count = data ['interval_count']
        virtual = trucks ['virtual']
        assignments = dummy_data (trucks, interval_time, interval_count, em_data)
        '''
        for i in range (len(trucks)):
            if (virtual [i] == True):
        '''       
        

def dummy_data (trucks, interval_time, interval_count, em_data):
    lat = trucks ['location']['lat']
    long = trucks ['location']['long']
    virtual = trucks ['virtual']

    data = np.hstack((lat, long))
    data = np.hstack((data, virtual))
    data = [[data[:0,], data[:1,], data[:2,]] for (lat, long, virtual) in data if virutal == True]
    data = np.array(data)

    kmeans = KMeans(n_clusters = len(data)).fit(data[:,0:2])
    centers = kmeans.cluster_centers_

    return centers




   
'''   
@app.route('/login', methods = ['POST', 'GET'])
def login():
    if (request.method == 'POST'):
        return redirect(url_for('/assignments'))
      
    else:
        start_time = request.form.get('start_time')
        interval_count = request.form.get('interval_count')
        return redirect(url_for('/emergencies', start_time = start_time, interval_count = interval_count))
'''
if __name__ == '__main__':
   app.run()
