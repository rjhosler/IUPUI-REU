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
import wasserstein
import PointProcess
from wasserstein import Cluster
from PointProcess import PointProcessTrain
from PointProcess import PointProcessRun
import json
from bson import json_util
import time

from flask import Flask, redirect, url_for, request, jsonify     
application = Flask(__name__)

PointProcess = PointProcessRun(param_location = 'Trained_Params_.npz')

@application.route('/emergencies')
def emergencies():
    start_time = request.args.get('start_time')
    interval_count = request.args.get('interval_count')
    interval_count = int(interval_count)
    if (request.args.get('time_interval')):
        time_interval = request.args.get('time_interval')
        time_interval = int(time_interval)
    else:
        time_interval = 15
 
    start_time =  datetime.datetime.fromtimestamp(float(start_time))
    total_output = []

    predictions, times, increment = PointProcess.get_events_for_api(start_time, interval_count, top_percent = 0)

    pred_max = 0
    for j in range (int(interval_count)):            
        output = {
            'start': times[j],
            'interval_length': increment,
            'emergencies': []
        }
        for i in range (len(predictions[0])):
            output ['emergencies'].append({
                'intensity': predictions[j][i][2],
                'location': {
                    'lat': predictions[j][i][0],
                    'long': predictions[j][i][1]
                    }
            })
            
            if (predictions[j][i][2] > pred_max):
                pred_max = predictions[j][i][2]
                im_save = {
                'intensity': predictions[j][i][2],
                'location': {
                    'lat': predictions[j][i][0],
                    'long': predictions[j][i][1]
                    }
                }
            
        total_output.append (output)
        print (len(output['emergencies']))

        with open('GET.json', 'w') as fp:
            json.dump(im_save, fp)

    return jsonify(total_output)


@application.route('/SingleProcessUpdate')
def SingleProcessUpdate():
    '''
    http://127.0.0.1:5000/SingleProcessUpdate?xcoord=-86.43&ycoord=39.14&timestamp=1532959162
    info should have form of: xcoord, ycoord, unix timestamp
    '''
    xcoord = [float(request.args.get('xcoord'))]
    ycoord = [float(request.args.get('ycoord'))]
    datetime = request.args.get('timestamp')
    datetime = [datetime.datetime.fromtimestamp(float(datetime))]
    update_df = {'XCOORD': xcoord, 'YCOORD': ycoord, 'DATE_TIME': datetime}
    update_df = pd.DataFrame(update_df)
    msg = PointProcess.update_from_new_inputs(update_df)

    return msg

@application.route('/ProcessUpdate/<name>')
def ProcessUpdate(name):
    fields = ['XCOORD', 'YCOORD', 'CALL_TYPE_FINAL_D', 'CALL_TYPE_FINAL', 'DATE_TIME']
    name = pd.read_csv(name, usecols=fields)
    name['DATE_TIME'] =  pd.to_datetime(name['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')
    name = name.sort_values(by='DATE_TIME')
    msg = PointProcess.update_from_new_inputs(name)

    return msg

@application.route('/login', methods = ['POST', 'GET'])
def login():
   if (request.method == 'POST'):
      user = request.form.get('nm', None)
      return redirect(url_for('ProcessUpdate',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('ProcessUpdate',name = user))

@application.route('/assignments', methods = ['POST'])
def assignments():
    if (request.method == 'POST'):
    
        data = request.get_json()
        
        trucks = filter_data(data ['trucks'])
        start_time = datetime.datetime.fromtimestamp(float(data ['start_time']))
        interval_time = data ['interval_time']
        interval_count = data ['interval_count']
        virtual = trucks [:,2]
        
        assignments = wasserstein_cluster (trucks, interval_time, interval_count, start_time)

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

#practical assignments
'''
    em_data is the locs for wasserstein over the given interval
    2.) Call Cluster in wasserstein passing em_data and len(all trucks with virtual = true)
    3.) Initialize centers with the locations of the true trucks
    4.) Cluster the data and return the centers
'''
def wasserstein_cluster (trucks, interval_time, interval_count, start_time):
    move_data, still_data = shrink_data (trucks)
    grid_loc = PointProcess.locs_for_wasserstein (start_time = start_time, num_projections = interval_count, top_percent = 80)
    end_time = start_time + datetime.timedelta(seconds = 15*60*interval_count)

    cluster = Cluster (grid_loc, len(move_data))
    cluster.set_centers (move_data[:,0:2], len(move_data))
    #if (still_data.size > 0):
    #    cluster.remove_points (still_data [:,0:2])

    lam = cluster.learn_lam(5, False, len (grid_loc))
    #cluster.round_off()
    data = grid_loc
    bincount = 90

    #Wasserstein
    centers = cluster.get_centers()
    dist = cluster.get_dist()
    
    centers = close_assignment (centers, move_data)    
    return centers

def shrink_data (trucks):
    data = np.zeros((len(trucks),3))
    data[:,0] = trucks [:,0]
    data[:,1] = trucks [:,1]
    data[:,2] = trucks [:,2]
    data = data.tolist()
    data1 = [[lat, long, virtual] for (lat, long, virtual) in data if virtual == True]
    data1 = np.array(data1)
    data2 = [[lat, long, virtual] for (lat, long, virtual) in data if virtual == False]
    data2 = np.array(data2)
    return data1, data2

#have trucks go to the assigned location that is closest to them
def close_assignment (centers, trucks):
    mindist = 9999
    pos = 0
    for i in range (len(trucks)):
        for j in range (len(centers)):
            dist = la.norm (trucks [i, 0:2] - centers [j, :])
            if (j == 0 or dist < mindist):
                mindist = dist
                pos = j
        if (i > 0):
            ordered_centers = np.append (ordered_centers, centers [pos, :], axis = 0)
        else:
            ordered_centers = np.copy(centers [pos, :])
        centers = np.delete (centers, pos, 0)
    ordered_centers = ordered_centers.reshape((len(ordered_centers) // 2, 2))
    return ordered_centers

if __name__ == '__main__':
    application.run()
