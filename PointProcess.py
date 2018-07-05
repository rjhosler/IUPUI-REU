import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from math import *
import datetime as datetime
import warnings
from numpy import unravel_index
from numpy import array
import time
import pylab as pl
from IPython import display

class PointProcessTrain:
    #training_points is a dataframe with labels: DATE_TIME (datetime format), XCOORD, YCOORD
    #xmin,xmax,ymin,ymax are in lat and long and initialized for Marion county
    #omega (w) is initialized to generic points
    #time_step selects multiplier for dt
    def __init__ (self, training_points, xgridsize = 100, ygridsize = 100, 
        xmin = -86.4619147125, xmax =  -85.60543100000002, ymin = 39.587905, ymax = 40.0099, 
        w = [.5, .1, .05], time_scale_label = 'days', pred_interval_label = '15minutes', update_with_trends = False, 
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
        self._LastTime = self._data.DATE_TIME[0]

        self._save_out = save_loc

        self._time_scale_label = time_scale_label
        self._time_scaling_lookup = {'days': 1.15741e-5} # convert delta seconds to delta days
        self._time_scale = self._time_scaling_lookup[self._time_scale_label]

        self._day = np.ones(7)*1/7
        self._pred_interval_label = pred_interval_label
        self._hour_vector_subdivision_lookup = {'hours': 1, '15minutes': 4, 'minutes': 60}
        self._hour_subdivision = self._hour_vector_subdivision_lookup[pred_interval_label]
        self._hour = np.ones(24*self._hour_subdivision)*1/(24*self._hour_subdivision)

        self._update_with_trends = update_with_trends

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

    def get_hour_vec_indx(self, event_time):
        return event_time.hour*self._hour_subdivision + floor(event_time.minute/(60/self._hour_subdivision))

    def param_update(self, event_time, last_event_time, xcoord, ycoord, 
        day_prob, hour_prob,
        Lam, F, mu, theta, Gtimes):

        # place new event in correct grid
        gx, gy = self.coord_to_grid(xcoord, ycoord)  

        # find day and hour
        curr_day = event_time.weekday()
        curr_hour = self.get_hour_vec_indx(event_time)

        # find global time delta 
        time_delta = (event_time - last_event_time).total_seconds()*self._time_scale
        last_event_time = event_time

        # update periodic trends
        dt_day = 0.0005
        for i in range(0, 7):
            day_prob[i] = (1-dt_day)*day_prob[i]
        day_prob[curr_day] += dt_day
        dt_hour = .000005
        for i in range(0, 24):
            hour_prob[i] = (1-dt_hour)*hour_prob[i]
        hour_prob[curr_hour] += dt_hour

        # global update of all grids: decay _F and get intensity
        for x in range(0, self._xsize):
            for y in range(0, self._ysize):
                for k in range(0, self._K):
                    F[x][y][k] = F[x][y][k] * np.exp(-1*self._w[k]*time_delta)
                Lam[x][y] = self.get_intensity(mu[x][y], sum(F[x][y]), hour_prob[curr_hour], day_prob[curr_day], time_weighted = True)

        # local update based on where event occurred
        dt = 0.005
        g_time_delta = (event_time - Gtimes.at[gx,gy]).total_seconds()*self._time_scale
        Gtimes.at[gx,gy] = event_time

        Lam_g = self.get_intensity(mu[gx][gy], sum(F[gx][gy]), hour_prob[curr_hour], day_prob[curr_day], time_weighted = self._update_with_trends)
        if Lam_g == 0:
            Lam_g = 1e-70
        mu[gx][gy] = mu[gx][gy] + dt * (mu[gx][gy]/Lam_g - mu[gx][gy] * g_time_delta)
        for k in range(0, self._K):
            theta[k] = theta[k] + dt * (F[gx][gy][k]/Lam_g - theta[k])
            F[gx][gy][k] = F[gx][gy][k] + self._w[k]*theta[k]

        return day_prob, hour_prob, Lam, F, mu, theta, Gtimes, last_event_time

    def get_intensity(self, mu_xy, sum_F_xy, hour_prob, day_prob, time_weighted):
        # get intensity. time_weighted = boolean 
        if time_weighted and not self._update_with_trends:
            # If model parameters are not normally calculated with trends factored in, need to scale day_prob so Lambda comes out in #/hour_subdivision.
            day_prob = day_prob * 7
            Lam_xy = (mu_xy + sum_F_xy)*hour_prob*day_prob
        elif time_weighted and self._update_with_trends:
            # If model parameters are calculated with trends, this is the way that all values of lambda are calculated. No need to do any scaling.
            Lam_xy = (mu_xy + sum_F_xy)*hour_prob*day_prob
        elif not time_weighted:
            # This is for calculating model parameters without factoring trends in. 
            Lam_xy = (mu_xy + sum_F_xy)
        return Lam_xy

    def train(self):
        for i in range(1, self._data_length):

            self._F[i] = self._F[i-1]
            self._mu[i] = self._mu[i-1]
            self._theta[i] = self._theta[i-1]

            self._day, self._hour, self._Lam[i], self._F[i], self._mu[i], self._theta[i], self._Gtimes, self._LastTime = self.param_update(
                self._data.DATE_TIME[i], self._data.DATE_TIME[i-1], self._data.XCOORD[i], self._data.YCOORD[i],
                self._day, self._hour,
                self._Lam[i], self._F[i], self._mu[i], self._theta[i], self._Gtimes)
        
        self.save_params()

    def save_params(self, memory_length = 10):

        if len(self._Lam) < memory_length:
            memory_length = len(self._Lam)

        Lam_save = self._Lam[-memory_length:,:]
        theta_save = self._theta[-memory_length:,:]
        F_save = self._F[-memory_length:,:]
        mu_save = self._mu[-memory_length:,:]

        np.savez(self._save_out, Lam = Lam_save, theta = theta_save, w = self._w, 
            F = F_save, mu = mu_save, day_prob = self._day, hour_prob = self._hour,
            grid_times = self._Gtimes.values, time_scale = self._time_scale_label, 
            grid_info = [self._xsize, self._ysize, self._xmin, self._xmax, self._ymin, self._ymax],
            last_time = self._LastTime, pred_interval_hourly_subdivision = self._hour_subdivision, 
            update_with_trends = self._update_with_trends, 
            save_loc = self._save_out)

    def param_examine(self):
        for i in range(0, self._K):
            plt.plot(np.transpose(self._theta)[i], label="w = " + str(self._w[i]))
        plt.title("data points vs. theta")
        plt.legend()
        plt.show()

        print("\nHour vector: ")
        print(self._hour)
        print("Hour vector sum: ")
        print(sum(self._hour))
        print("\nDay vector: ")
        print(self._day)
        print("Day vector sum: ")
        print(sum(self._day))

    def model_hotspot_examine(self, num_points, num_hotspots = 10):
        # examine how top model cells compare to actual top cells over data used for training

        start = len(self._Lam) - num_points
        if start < 0:
            start = 0
        end = len(self._Lam)-1

        sum_intensity = sum(self._Lam[start:,:])
        print("Location and value of largest and smallest sum(Lambda): ")
        print(np.amax(sum_intensity), unravel_index(sum_intensity.argmax(), sum_intensity.shape),
            np.amin(sum_intensity), unravel_index(sum_intensity.argmin(), sum_intensity.shape))

        tot_events = np.zeros([self._xsize, self._ysize])
        for i in range(start, end):
            x, y = self.coord_to_grid(self._data.XCOORD[i], self._data.YCOORD[i])
            tot_events[x][y] = tot_events[x][y] + 1
        print("\nLocation of grid with most events:")
        print(np.amax(tot_events), unravel_index(tot_events.argmax(), tot_events.shape))

        sum_intensity_copy = np.copy(sum_intensity)
        pred_locs = []

        tot_events_copy = np.copy(tot_events)
        actual_locs = []

        for i in range(0, num_hotspots):
            indx = unravel_index(tot_events_copy.argmax(), tot_events_copy.shape)
            actual_locs.append(indx)
            tot_events_copy[indx[0]][indx[1]] = 0

        for i in range(0, num_hotspots):
            indx = unravel_index(sum_intensity_copy.argmax(), sum_intensity_copy.shape)
            pred_locs.append(indx)
            sum_intensity_copy[indx[0]][indx[1]] = 0

        print("\nHistorical time period is " + str((self._data.DATE_TIME[end] - self._data.DATE_TIME[start]).total_seconds()*self._time_scale) + " " + str(self._time_scale_label))

        print("\nTrained hotspots in real top " + str(num_hotspots))
        for i in range(0, num_hotspots):
            if pred_locs[i] in actual_locs:
                x = pred_locs[i][0]
                y = pred_locs[i][1]
                predicted_number = sum_intensity[x][y]*(1/num_points*(self._data.DATE_TIME[end]-self._data.DATE_TIME[start]).total_seconds()*self._time_scaling_lookup[self._pred_interval_label])
                print("Grid: " + str(pred_locs[i]) +", Model: "+ str(predicted_number)+", Real: "+ str(int(tot_events[x][y])) )

        print("\nTrained hotstpots not in real top " + str(num_hotspots))
        for i in range(0, num_hotspots):
            if pred_locs[i] not in actual_locs:
                x = pred_locs[i][0]
                y = pred_locs[i][1] 
                predicted_number = sum_intensity[x][y]*(1/num_points*(self._data.DATE_TIME[end]-self._data.DATE_TIME[start]).total_seconds()*self._time_scaling_lookup[self._pred_interval_label])
                print("Grid: " + str(pred_locs[i]) +", Model: "+ str(predicted_number)+", Real: "+ str(int(tot_events[x][y])) )

        print("\nReal top " + str(num_hotspots) +" hotspots not in trained model")
        for i in range(0, num_hotspots):
            if actual_locs[i] not in pred_locs:
                x = actual_locs[i][0]
                y = actual_locs[i][1]
                predicted_number = sum_intensity[x][y]*(1/num_points*(self._data.DATE_TIME[end]-self._data.DATE_TIME[start]).total_seconds()*self._time_scaling_lookup[self._pred_interval_label])
                print("Grid: " + str(actual_locs[i]) +", Model: "+ str(predicted_number)+", Real: "+ str(int(tot_events[x][y])) )

   
class PointProcessRun(PointProcessTrain):

    def __init__(self, param_location = 'Trained_Params.npz'):

        trained_params = np.load(param_location)
        self._Lam = trained_params['Lam']
        self._theta = trained_params['theta']
        self._w = trained_params['w']
        self._F = trained_params['F']
        self._mu = trained_params['mu']
        self._day = trained_params['day_prob']
        self._hour = trained_params['hour_prob']
        self._Gtimes = pd.DataFrame(trained_params['grid_times'])
        self._LastTime = trained_params['last_time']
        self._K = len(self._w)

        self._save_out = str(trained_params['save_loc'])

        self._time_scale_label = str(trained_params['time_scale'])
        self._time_scaling_lookup = {'days': 1.15741e-5, 'hours': 0.0002777784, '15minutes': 0.001111111, 'minutes': 0.016666704, 'seconds': 1}
        self._time_scale = self._time_scaling_lookup[self._time_scale_label]

        self._hour_subdivision = int(trained_params['pred_interval_hourly_subdivision'])

        self._xsize = int(trained_params['grid_info'][0])
        self._ysize = int(trained_params['grid_info'][1])
        self._xmin = float(trained_params['grid_info'][2])
        self._xmax = float(trained_params['grid_info'][3])
        self._ymin = float(trained_params['grid_info'][4])
        self._ymax = float(trained_params['grid_info'][5])

        self._update_with_trends = trained_params['update_with_trends']

    def update_from_new_inputs(self, update_csv):
        # update_csv should have headers and format: DATE_TIME (datetime string, format='%Y-%m-%d %H:%M:%S'), XCOORD (longitude), YCOORD (latitude)
        
        update_data = pd.read_csv(update_csv)
        new_points = len(update_data)
        update_data.DATE_TIME = pd.to_datetime(update_data.DATE_TIME, format='%Y-%m-%d %H:%M:%S')
        update_data = update_data.sort_values(by = 'DATE_TIME')

        # Lam, F, mu and theta all have "memory". We work with the final index of each, but the last ~10 or so are saved for debugging purposes
        new_theta = np.zeros([new_points, self._K])
        new_Lam = np.zeros([new_points, self._xsize, self._ysize])
        new_mu = np.zeros([new_points, self._xsize, self._ysize])
        new_F = np.zeros([new_points, self._xsize, self._ysize, self._K])

        for i in range(0, new_points):
            self._day, self._hour, new_Lam[i], new_F[i], new_mu[i], new_theta[i], self._Gtimes, self._LastTime = self.param_update(
                update_data.DATE_TIME[i], self._LastTime, update_data.XCOORD[i], update_data.YCOORD[i],
                self._day, self._hour,
                self._Lam[-1], self._F[-1], self._mu[-1], self._theta[-1], self._Gtimes)

        self._Lam = np.concatenate((self._Lam, new_Lam), axis=0)
        self._F = np.concatenate((self._F, new_F), axis=0)
        self._mu = np.concatenate((self._mu, new_mu), axis=0)
        self._theta = np.concatenate((self._theta, new_theta), axis=0)

        self._Lam = self._Lam[1:,:]
        self._F = self._F[1:,:]
        self._mu = self._mu[1:,:]
        self._theta = self._theta[1:,:]

        self.save_params()

    def calculate_future_intensity(self, future_time):  
        # calls get_intensity to get each indivicual value of Lamba

        time_delta = (future_time - self._LastTime).total_seconds()*self._time_scale

        future_hour = self.get_hour_vec_indx(future_time) 
        future_day =  future_time.weekday()

        decayed_F_xy = np.zeros(3)

        pred_Lam = np.zeros([self._xsize, self._ysize])
        for x in range(0, self._xsize):
            for y in range(0, self._ysize):
                for k in range(0, self._K):
                    decayed_F_xy[k] = self._F[-1][x][y][k]*np.exp(-self._w[k]*time_delta)  
                pred_Lam[x][y] = self.get_intensity(self._mu[-1][x][y], sum(decayed_F_xy), self._hour[future_hour], self._day[future_day], time_weighted = True)
        return pred_Lam

    def get_future_events(self, start_time, num_periods, reshape = False):
        # calls calculate_future_intensity to find intensity matrix @ each time interval. Returns matrix format, times array and format of [xcoord, ycoord, intensity]
        times = []
        time_increment = self.get_time_increment()         # time increment is dependent on how the hour vector is subdivided
        intensity_predictions = np.zeros([num_periods, self._xsize, self._ysize])

        for i in range(0, num_periods):
            future_time = start_time + datetime.timedelta(seconds = time_increment*i)
            times.append(future_time)
            intensity = self.calculate_future_intensity(future_time)   
            intensity_predictions[i] = intensity

        if reshape:
            intensity_predictions_reshaped = np.zeros([num_periods, self._xsize*self._ysize, 3])
            for i in range(0, num_periods):
                intensity_predictions_reshaped[i] = self.reshape_lam(intensity_predictions[i])
            intensity_predictions = np.copy(intensity_predictions_reshaped)

        return intensity_predictions, array(times), time_increment

    def get_time_increment(self):
        return (1 / self._hour_subdivision)/self._time_scaling_lookup['hours'] 

    def test_projection(self, test_points, num_hotspots = 10, plot = False):
        # test points is a data frame with labels DATE_TIME (datetime format), XCOORD, YCOORD
        time_period = (test_points.DATE_TIME[len(test_points)-1] - test_points.DATE_TIME[0]).total_seconds()

        time_increment = self.get_time_increment() 

        # get everything in seconds to find number of periods to model
        num_periods = ceil(time_period/time_increment)                                              # number of predictions to run at time_increment value each

        print("\nPredicting over time of " + str(time_period*self._time_scale) + " " + str(self._time_scale_label) + ". Generating " + str(num_periods) + " intensity prediction(s)")

        intensity_predictions, time_increments, time_increment_unit = self.get_future_events(test_points.DATE_TIME[0], num_periods)

        # sum to get prediction over total time of test_points
        pred_num_events = sum(intensity_predictions[:,:])

        # fiid location of num_hotspots predicted hotspots
        c_pred_num_events = np.copy(pred_num_events)
        pred_locs = []
        for i in range(0, num_hotspots):
            indx = unravel_index(c_pred_num_events.argmax(), c_pred_num_events.shape)
            pred_locs.append(indx)
            c_pred_num_events[indx[0]][indx[1]] = 0    

        # get total number of events in each grid 
        tot_events = np.zeros([self._xsize, self._ysize])
        for i in range(0, len(test_points)):
            x, y = self.coord_to_grid(test_points.XCOORD[i], test_points.YCOORD[i])
            tot_events[x][y] = tot_events[x][y] + 1

        # find the location of num_hotspots actual hotspots
        c_tot_events= np.copy(tot_events)
        actual_locs = []
        for i in range(0, num_hotspots):
            indx = unravel_index(c_tot_events.argmax(), c_tot_events.shape)
            actual_locs.append(indx)
            c_tot_events[indx[0]][indx[1]] = 0

        print("\nPredicted hotspots in real top " + str(num_hotspots))
        for i in range(0, num_hotspots):
            if pred_locs[i] in actual_locs:
                x = pred_locs[i][0]
                y = pred_locs[i][1]
                print("Grid: " + str(pred_locs[i]) +", Model: "+ str(pred_num_events[x][y]) + ", Real: " + str(tot_events[x][y]))

        print("\nPredicted hotstpots not in real top " + str(num_hotspots))
        for i in range(0, num_hotspots):
            if pred_locs[i] not in actual_locs:
                x = pred_locs[i][0]
                y = pred_locs[i][1]
                print("Grid: " + str(pred_locs[i]) +", Model: "+ str(pred_num_events[x][y]) + ", Real: " + str(tot_events[x][y]))

        print("\nReal top " +str(num_hotspots) +"  hotspots not predicted")
        for i in range(0, num_hotspots):
            if actual_locs[i] not in pred_locs:
                x = actual_locs[i][0]
                y = actual_locs[i][1]
                print("Grid: " + str(actual_locs[i]) +", Model: "+ str(pred_num_events[x][y]) + ", Real: " + str(tot_events[x][y]))

        if plot:
            plt.figure(figsize=(20,10))
            displays = 50
            interval = ceil(num_periods/50)
            for n in range(0, displays): 
                i = n*interval
                if i < num_periods:
                    plt.title('Events in window surrounding time: '+time_increments[i].strftime('%Y-%m-%d %H:%M:%S'))
                    plt.imshow(intensity_predictions[i], cmap = 'hot', interpolation = 'nearest')
                    display.clear_output(wait=True)
                    display.display(pl.gcf())
                    time.sleep(.0005)
                elif i >= num_periods:
                    break 

        return intensity_predictions, time_increments

    def locs_for_wasserstein(self, start_time, num_projections = 16):

        predictions, times, time_increment_unit = self.get_future_events(start_time, num_projections)

        sum_predictions = sum(predictions[:,:])     # use matrix form for summation.
        pred_val_lst = sum_predictions.reshape(self._xsize*self._ysize//1).tolist()
        mode = max(set(pred_val_lst), key = pred_val_lst.count)

        reshaped_sum = self.reshape_lam(sum_predictions) 

        condensed = np.empty((0,0,0))

        for i in range(0, len(reshaped_sum)):
            if reshaped_sum[i][2] != mode and reshaped_sum[i][2] > 0:
                condensed = np.append(condensed, reshaped_sum[i])
        # reshape to [xcoord, ycoord, lam]
        condensed = condensed.reshape((len(condensed)//3,3))

        return condensed

    def reshape_lam(self, lam):
        # reshapes matrix of intensities into list of: xcord, ycord, intensity
        x_y_lam = np.empty((0,0,0))
        for x in range(0, self._xsize):
            for y in range(0, self._ysize):
                xcoord, ycoord = self.grid_to_coord(x, y)
                # format is latitude, longitude, intensity
                to_append = ycoord, xcoord, lam[x][y]
                x_y_lam = np.append(x_y_lam, to_append)

        x_y_lam = x_y_lam.reshape ((len(x_y_lam)//3,3))
        return x_y_lam


