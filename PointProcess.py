import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from math import *
import datetime as datetime
import warnings
from numpy import unravel_index

class PointProcessTrain:
    #training_points is a dataframe with labels: DATE_TIME (datetime format), XCOORD, YCOORD
    #xmin,xmax,ymin,ymax are in lat and long and initialized for Marion county
    #omega (w) is initialized to generic points
    #time_step selects multiplier for dt
    def __init__ (self, training_points, xgridsize = 100, ygridsize = 100, 
        xmin = -86.4619147125, xmax =  -85.60543100000002, ymin = 39.587905, ymax = 40.0099, 
        w = [.5, .1, .05], time_scale_label = 'days', 
        save_loc = 'Trained_Params.npz'):
        
        self._data = training_points 
        self._data_length = len(self._data) 
        self._xsize = xgridsize
        self._ysize = ygridsize
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

        self._w = w
        self._K = len(w)
        self._mu = np.ones([len(self._data), self._xsize, self._ysize])*0.0000001
        self._F = np.ones([len(self._data), self._xsize, self._ysize, self._K])*.1
        self._Lam = np.ones([len(self._data), self._xsize, self._ysize])*0.000000001
        self._theta = np.ones([len(self._data), self._K])*.1
        self._Gtimes = pd.DataFrame(np.zeros([xgridsize, ygridsize]))
        self._Gtimes[:] = self._data.DATE_TIME[0]

        self._hour = np.ones(24)
        self._day = np.ones(32)

        self._save_out = save_loc

        self._time_scale_label = time_scale_label
        time_scaling_lookup = {'days': 1.15741e-5, 'hours': 0.0002777784, 'minutes': 0.016666704, 'seconds': 1}
        self._time_scale = time_scaling_lookup[self._time_scale_label]

    def coord_to_grid(self, xcoord, ycoord):
        if xcoord < self._xmin:
            xcoord = self._xmin
        elif xcoord > self._xmax:
            xcoord = self._xmax

        if ycoord < self._ymin:
            ycoord = self._ymin
        elif ycoord > self._ymax:
            ycoord = self._ymax  

        xbin = int((xcoord-self._xmin)/(self._xmax-self._xmin)*(self._xsize-1))
        ybin = int((ycoord-self._ymin)/(self._ymax-self._ymin)*(self._ysize-1))
        return xbin, ybin

    def grid_to_coord(self, xbin, ybin):
        xcoord = (xbin * (self._xmax - self._xmin)) / (self._xsize -1) + self._xmin
        ycoord = (ybin * (self._ymax - self._ymin)) / (self._ysize -1) + self._ymin
        return xcoord, ycoord

    def update(self, event_time, last_event_time, xcoord, ycoord, index):

        self._mu[index] = self._mu[index-1]
        self._F[index] = self._F[index-1]
        self._theta[index] = self._theta[index-1]


        # place new event in correct grid
        gx, gy = self.coord_to_grid(xcoord, ycoord)  

        # find day and hour
        curr_day = event_time.day
        curr_hour = event_time.hour 

        # find global time delta 
        time_delta = (event_time - last_event_time).total_seconds()*self._time_scale

        # update periodic trends
        dt_day = .05
        for i in range(0, 32):
            self._day[i] = (1-dt_day)*self._day[i]
        self._day[curr_day] += dt_day
        dt_hour = .005
        for i in range(0, 24):
            self._hour[i] = (1-dt_hour)*self._hour[i]
        self._hour[curr_hour] += dt_hour 

        # global update of all grids
        for x in range(0, self._xsize):
            for y in range(0, self._ysize):
                for k in range(0, self._K):
                    self._F[index][x][y][k] = self._F[index][x][y][k] * np.exp(-1*self._w[k]*time_delta)
                self._Lam[index][x][y] = self._mu[index][x][y] + sum(self._F[index][x][y])*self._hour[curr_hour]*self._day[curr_day]

        # local update based on where event occurred
        dt = 0.005
        g_time_delta = (event_time - self._Gtimes.at[gx,gy]).total_seconds()*self._time_scale
        self._Gtimes.at[gx,gy] = event_time
        if self._Lam[index][gx][gy] == 0:
            self._Lam[index][gx][gy] = 1e-70
        self._mu[index][gx][gy] = self._mu[index][gx][gy] + dt * (self._mu[index][gx][gy]/self._Lam[index][gx][gy] * g_time_delta)
        for k in range(0, self._K):
            self._theta[index][k] = self._theta[index][k] + dt * (self._F[index][gx][gy][k]/self._Lam[index][gx][gy] - self._theta[index][k])
            self._F[index][gx][gy][k] = self._F[index][gx][gy][k] + self._w[k]*self._theta[index][k]

    def train(self):
        for i in range(1, self._data_length):
            self.update(self._data.DATE_TIME[i], self._data.DATE_TIME[i-1], self._data.XCOORD[i], self._data.YCOORD[i], i)
        np.savez(self._save_out, Lam = self._Lam, theta = self._theta, w = self._w, 
            F = self._F, mu = self._mu, day_prob = self._day, hour_prob = self._hour,
            grid_times = self._Gtimes.values)

    def param_examine(self, num_points, num_top_grids = 10):
        theta_track = []
        for i in range(0, self._data_length):
            theta_track.append(sum(self._theta[i]))
        plt.title("approx theta vs. data points")
        plt.plot(theta_track)

        start = self._data_length - num_points
        end = self._data_length-1

        print("Time period is " + str((self._data.DATE_TIME[end] - self._data.DATE_TIME[start]).total_seconds()*self._time_scale) + str(self._time_scale_label))

        print("\nMax Lambda, Loc of Max Lambda, Min Lambda, Loc of Min Lambda")
        summed_intensity = sum(self._Lam[start:,:])
        print(np.amax(summed_intensity), unravel_index(summed_intensity.argmax(), summed_intensity.shape),
            np.amin(summed_intensity), unravel_index(summed_intensity.argmin(), summed_intensity.shape))

        print("\nMost events in grid, Location of grid with most events")
        total_events = np.zeros([self._xsize, self._ysize])
        for i in range(start, end):
            x, y = self.coord_to_grid(self._data.XCOORD[i], self._data.YCOORD[i])
            total_events[x][y] = total_events[x][y] + 1
        print(np.amax(total_events), unravel_index(total_events.argmax(), total_events.shape))

        print("\nHour vector: ")
        print(self._hour)
        print("Day vector: ")
        print(self._day)

        # examine how top predicted cells compare to actual top cells
        tot_intensity_copy = np.copy(summed_intensity)
        pred_locs = []

        tot_events_copy = np.copy(total_events)
        actual_locs = []

        for i in range(0, num_top_grids):
            indx = unravel_index(tot_events_copy.argmax(), tot_events_copy.shape)
            actual_locs.append(indx)
            tot_events_copy[indx[0]][indx[1]] = 0

        for i in range(0, num_top_grids):
            indx = unravel_index(tot_intensity_copy.argmax(), tot_intensity_copy.shape)
            pred_locs.append(indx)
            tot_intensity_copy[indx[0]][indx[1]] = 0

        print("\nTop model hotspots in real top 10:")
        for i in range(0, num_top_grids):
            if pred_locs[i] in actual_locs:
                predicted_number = summed_intensity[pred_locs[i][0]][pred_locs[i][1]]*(1/num_points*(self._data.DATE_TIME[end]-self._data.DATE_TIME[start]).total_seconds()*self._time_scale)
                print("Grid: " + str(pred_locs[i]) +", Model: "+ str(predicted_number)+", Real: "+ str(int(total_events[pred_locs[i][0]][pred_locs[i][1]])))

        print("\nTop model hotstpots not in real top 10")
        for i in range(0, num_top_grids):
            if pred_locs[i] not in actual_locs:
                predicted_number = summed_intensity[pred_locs[i][0]][pred_locs[i][1]]*(1/num_points*(self._data.DATE_TIME[end]-self._data.DATE_TIME[start]).total_seconds()*self._time_scale)
                print("Grid: " + str(pred_locs[i]) +", Model: "+ str(predicted_number)+", Real: "+ str(int(total_events[pred_locs[i][0]][pred_locs[i][1]])))

        print("\nReal top 10 hotspots not predicted by model")
        for i in range(0, num_top_grids):
            if actual_locs[i] not in pred_locs:
                predicted_number = summed_intensity[actual_locs[i][0]][actual_locs[i][1]]*(1/num_points*(self._data.DATE_TIME[end]-self._data.DATE_TIME[start]).total_seconds()*self._time_scale)
                print("Grid: " + str(actual_locs[i]) +", Model: "+ str(predicted_number)+", Real: "+ str(int(total_events[actual_locs[i][0]][actual_locs[i][1]])))


class PointProcess:
    def __init__(self):
        pass