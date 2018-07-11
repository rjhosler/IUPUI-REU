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
        xmin = -86.3283, xmax =  -85.8942, ymin = 39.6277, ymax = 39.9277, 
        w = [.5, .1, .05], pred_interval_label = '15minutes', update_with_trends = False, track_granularity = 1000, lam_memory = 500,
        final_param_save_loc = 'Trained_Params.npz', param_track_save_loc = 'Track_Of_Params.npz'):
        
        self._data = training_points 
        self._data_length = len(self._data) 
        self._xsize = xgridsize
        self._ysize = ygridsize
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

        self._w = array(w)
        self._K = len(w)
        self._mu = np.ones([self._xsize, self._ysize])*0.02
        self._F = np.ones([self._xsize, self._ysize, self._K])*.1
        self._Lam = np.ones([self._xsize, self._ysize])*0.001
        self._theta = np.ones([self._K])*.1
        self._Gtimes = pd.DataFrame(np.zeros([xgridsize, ygridsize]))
        self._Gtimes[:] = self._data.DATE_TIME[0]
        self._LastTime = self._data.DATE_TIME[0]

        self._save_out = final_param_save_loc
        self._track_out = param_track_save_loc

        self._track_granularity = track_granularity
        memory_length = floor(self._data_length/self._track_granularity)

        self._mu_track = np.ones([memory_length, self._xsize, self._ysize])
        self._Lam_track = np.ones([memory_length, self._xsize, self._ysize])
        self._F_track = np.ones([memory_length, self._xsize, self._ysize, self._K])
        self._theta_track = np.ones([memory_length, self._K])

        self._lam_memory = lam_memory
        self._Lam_for_hotspots = np.ones([lam_memory, self._xsize, self._ysize])

        self._time_scale_label = 'days'
        self._time_scaling_lookup = {'days': 1.15741e-5, 'hours': 0.0002777784, '15minutes': 0.001111111, 'minutes': 0.016666704, 'seconds': 1}  # for converting from seconds to days, hours, etc. 
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

    def global_update(self, event_time, last_event_time, day_prob, hour_prob, F):
        # F is in matrix form here

        # find day and hour
        curr_day = event_time.weekday()
        curr_hour = self.get_hour_vec_indx(event_time)

        # find global time delta 
        time_delta = (event_time - last_event_time).total_seconds()*self._time_scale
        last_event_time = event_time

        # update periodic trends
        dt_day = .0001
        day_prob = (1-dt_day)*day_prob
        day_prob[curr_day] += dt_day

        dt_hour = .0005
        hour_prob = (1-dt_hour)*hour_prob
        hour_prob[curr_hour] += dt_hour

        # decay F (in its array form)
        F = F*np.exp(-1*self._w*time_delta)

        return last_event_time, day_prob, hour_prob, F
    
    def local_update(self, Gtimes, event_time, day_prob, hour_prob, F, mu, theta, gx, gy):

        # find day and hour
        curr_day = event_time.weekday()
        curr_hour = self.get_hour_vec_indx(event_time)

        # local update based on where event occurred
        dt = 0.005
        g_time_delta = (event_time - Gtimes.at[gx,gy]).total_seconds()*self._time_scale
        Gtimes.at[gx,gy] = event_time

        Lam_g = self.get_intensity(mu[gx][gy], F[gx][gy], hour_prob[curr_hour], day_prob[curr_day], time_weighted = self._update_with_trends)
        if Lam_g == 0:
            Lam_g = 1e-70
        mu[gx][gy] = mu[gx][gy] + dt * (mu[gx][gy]/Lam_g - mu[gx][gy] * g_time_delta)

        theta = theta + dt * (F[gx][gy]/Lam_g - theta)

        F[gx][gy] = F[gx][gy] + self._w*theta

        return F, mu, theta, Gtimes

    def get_intensity(self, mu, F, hour_prob, day_prob, time_weighted):
        # get intensity. time_weighted = boolean 

        if len(F.shape) == self._K:
            sum_F = np.sum(F, axis = 2)

        elif len(F.shape) == 1:
            sum_F = np.sum(F)
        else:
            print("Shape of F not appropriate. It is: " + str(F.shape))

        if time_weighted and not self._update_with_trends:
            # If model parameters are not normally calculated with trends factored in, need to scale day_prob so Lambda comes out in #/hour_subdivision.
            day_prob = day_prob * 7
            Lam = (mu + sum_F)*hour_prob*day_prob
        elif time_weighted and self._update_with_trends:
            # If model parameters are calculated with trends, this is the way that all values of lambda are calculated. No need to do any scaling.
            Lam = (mu + sum_F)*hour_prob*day_prob
        elif not time_weighted and not self._update_with_trends:
            # This is for calculating model parameters without factoring trends in. 
            Lam = mu + sum_F
        else:
            print("Error with weighting & dimensional anaysis")
        return Lam

    def train(self, progress_bar = True):
        for i in range(1, self._data_length):

            # place new event in correct grid
            gx, gy = self.coord_to_grid(self._data.XCOORD[i], self._data.YCOORD[i])

            # global update
            self._LastTime, self._day, self._hour, self._F = self.global_update(self._data.DATE_TIME[i],
                self._data.DATE_TIME[i-1], self._day, self._hour, self._F) 

            # start keeping track of intensities near the end to evaluate model performance:
            diff = self._data_length - i
            if diff <= self._lam_memory:

                indx = abs(diff - self._lam_memory)

                curr_day = self._data.DATE_TIME[i].weekday()
                curr_hour = self.get_hour_vec_indx(self._data.DATE_TIME[i])

                self._Lam_for_hotspots[indx] = self.get_intensity(self._mu, self._F, self._hour[curr_hour], self._day[curr_day], time_weighted = True)

            # local update: 
            self._F, self._mu, self._theta, self._Gtimes = self.local_update(self._Gtimes, self._data.DATE_TIME[i],
                self._day, self._hour, self._F, self._mu, self._theta, gx, gy)

            # Keeping track of paramters over time:
            if i%self._track_granularity == 0:
                indx = floor(i/self._track_granularity)
                self._mu_track[indx] = self._mu
                self._theta_track[indx] = self._theta
                self._Lam_track[indx] = self._Lam
                self._F_track[indx] = self._F

                if progress_bar:
                    print(str(i/self._data_length*100) + " percent trained\n")
                
        self.save_params(save_tracked_params = True)

    def save_params(self, save_tracked_params = False):

        np.savez(self._save_out, Lam = self._Lam, theta = self._theta, w = self._w, 
            F = self._F, mu = self._mu, day_prob = self._day, hour_prob = self._hour,
            grid_times = self._Gtimes.values, time_scale = self._time_scale_label, 
            grid_info = [self._xsize, self._ysize, self._xmin, self._xmax, self._ymin, self._ymax],
            last_time = self._LastTime, pred_interval_hourly_subdivision = self._hour_subdivision, 
            update_with_trends = self._update_with_trends, 
            save_loc = self._save_out)

        if save_tracked_params:
            np.savez(self._track_out, Lam_track = self._Lam_track, mu_track = self._mu_track, F_track = self._F_track, theta_track = self._theta_track, las_Lams = self._Lam_for_hotspots)

    def param_examine(self):
        for i in range(0, self._K):
            plt.plot(np.transpose(self._theta_track)[i], label="w = " + str(self._w[i]))
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

        if num_points > self._lam_memory:
            num_points = self._lam_memory

        start = self._data_length - num_points
        if start < 0:
            start = 0
        end = self._data_length -1

        sum_intensity = sum(self._Lam_for_hotspots[-num_points:,:])
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

    def __init__(self, param_location = 'Trained_Params.npz', save_loc = None):

        trained_params = np.load(param_location)
        self._Lam = trained_params['Lam']
        self._theta = trained_params['theta']
        self._w = trained_params['w']
        self._F = trained_params['F']
        self._mu = trained_params['mu']
        self._day = trained_params['day_prob']
        self._hour = trained_params['hour_prob']
        self._Gtimes = pd.DataFrame(trained_params['grid_times'])
        self._LastTime = trained_params['last_time'].tolist()
        self._K = len(self._w)

        if save_loc:
            self._save_out = save_loc
        else:
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
        update_data.reset_index(drop = True, inplace = True)

        indx = None
        for i in range(0, new_points):
            if update_data.DATE_TIME[i] > self._LastTime:
                indx = i
                break

        if indx is None:
            msg = 'No inputted events occurred at a later date than events previously used for training'
            return msg

        update_data = update_data[indx:]
        update_data.reset_index(drop = True, inplace = True)
        new_points = len(update_data)

        
        for i in range(0, new_points):

            # place new event in correct grid
            gx, gy = self.coord_to_grid(update_data.XCOORD[i], update_data.YCOORD[i])

            # global update
            self._LastTime, self._day, self._hour, self._F = self.global_update(update_data.DATE_TIME[i],
                self._LastTime, self._day, self._hour, self._F) 

            # local update: 
            self._F, self._mu, self._theta, self._Gtimes = self.local_update(self._Gtimes, update_data.DATE_TIME[i],
                self._day, self._hour, self._F, self._mu, self._theta, gx, gy)

        self.save_params()

        msg = 'Parameters updated: ' + str(new_points) + ' used for update ranging from: '+ update_data.DATE_TIME[0].strftime('%Y-%m-%d %H:%M:%S') + ' to ' + update_data.DATE_TIME[new_points-1].strftime('%Y-%m-%d %H:%M:%S')+'.'
        return msg

    def calculate_future_intensity(self, future_time, decay = False):  
        # calls get_intensity to get each indivicual value of Lamba

        time_delta = (future_time - self._LastTime).total_seconds()*self._time_scale

        future_hour = self.get_hour_vec_indx(future_time) 
        future_day =  future_time.weekday()

        if decay:
            F = self._F*np.exp(-1*self._w*time_delta)
        else:
            F = self._F
        pred_Lam = self.get_intensity(self._mu, F, self._hour[future_hour], self._day[future_day], time_weighted = True)

        return pred_Lam

    def get_future_events(self, start_time, num_periods, top_percent):
        # calls calculate_future_intensity to find intensity matrix @ each ti,me interval. Returns matrix format, times array and format of [xcoord, ycoord, intensity]
        # top_percent is lowest percentile to keep in the data set
        times = []
        time_increment = self.get_time_increment()         # time increment is dependent on how the hour vector is subdivided
        intensity_predictions = np.zeros([num_periods, self._xsize, self._ysize])

        if top_percent:
            if top_percent > 100:
                top_percent = 100
            if top_percent < 0:
                top_percent = 0

        for i in range(0, num_periods):
            future_time = start_time + datetime.timedelta(seconds = time_increment*i)
            times.append(future_time)
            intensity = self.calculate_future_intensity(future_time) 

            if top_percent:
                threshold = np.percentile(intensity, top_percent)
                intensity = intensity - threshold    # values below threshold become negative
                neg_indxs = intensity < 0            # find indices of values below 0
                intensity[neg_indxs] = 0             # set negative values to 0

            intensity_predictions[i] = intensity

        return intensity_predictions, array(times), time_increment

    def get_events_for_api(self, start_time, num_periods, top_percent = 90):
        # formats future predictions for the api
        intensity_predictions, times, time_increment = self.get_future_events(start_time, num_periods, top_percent)

        reshaped_intensity_predictions = []

        for i in range(0, len(intensity_predictions)):
            reshaped_intensity_predictions.append(self.reshape_lam(intensity_predictions[i], list_format = 'list'))

        return reshaped_intensity_predictions, times, time_increment

    def get_time_increment(self):
        return (1 / self._hour_subdivision)/self._time_scaling_lookup['hours'] 

    def test_projection(self, test_points, num_hotspots = 10, top_percent = 90):
        # test points is a data frame with labels DATE_TIME (datetime format), XCOORD, YCOORD
        time_period = (test_points.DATE_TIME[len(test_points)-1] - test_points.DATE_TIME[0]).total_seconds()

        time_increment = self.get_time_increment() 

        # get everything in seconds to find number of periods to model
        num_periods = ceil(time_period/time_increment)                                              # number of predictions to run at time_increment value each

        print("\nPredicting over time of " + str(time_period*self._time_scale) + " " + str(self._time_scale_label) + ". Generating " + str(num_periods) + " intensity prediction(s)")

        intensity_predictions, time_increments, time_increment_unit = self.get_future_events(test_points.DATE_TIME[0], num_periods, top_percent)

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

        pai, n_frac, a_frac = self.predictive_accuracy(pred_num_events, pred_locs, tot_events)

        print("\nThe predictive accuracy index for " + str(num_hotspots) + " hotspots is: " + str(pai) +". \nHit number/Tot number: " + str(n_frac) + ". Hit area/Tot area: " + str(a_frac))

        print("\nThe predicted number of events is: " +str(sum(sum(pred_num_events))))

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

        return intensity_predictions, time_increments, pred_num_events, pred_locs, tot_events, actual_locs

    def predictive_accuracy(self, tot_pred, pred_locs, tot_events):
        hit_num = 0
        tot = sum(sum(tot_events))
        for i in range(0, len(pred_locs)):
            x = pred_locs[i][0]
            y = pred_locs[i][1]
            hit_num = hit_num + tot_events[x][y]
        print(hit_num)
        n_frac = hit_num/tot
        a_frac = len(pred_locs)/(tot_pred.shape[0]*tot_pred.shape[1])
        
        pai = n_frac/a_frac

        return pai, n_frac, a_frac

    def locs_for_wasserstein(self, start_time, num_projections = 16, top_percent = 90):

        predictions, times, time_increment_unit = self.get_future_events(start_time, num_projections, top_percent)
        sum_predictions = sum(predictions[:,:])
        reshaped_sum = self.reshape_lam(sum_predictions, list_format = 'np') 
        return reshaped_sum

    def reshape_lam(self, lam, list_format = 'np'):
        # reshapes matrix of intensities into list of: xcord, ycord, intensity
        # return numpy format for wasserstein clustering, list format for predictions (because that allows for different lengths)
        # removes intensity values of 0

        x_y_lam = np.empty((0,0,0))
        for x in range(0, self._xsize):
            for y in range(0, self._ysize):
                xcoord, ycoord = self.grid_to_coord(x, y)
                # format is latitude, longitude, intensity
                if lam[x][y] > 0:
                    to_append = ycoord, xcoord, lam[x][y]
                    x_y_lam = np.append(x_y_lam, to_append)

        x_y_lam = x_y_lam.reshape ((len(x_y_lam)//3,3))

        if list_format != 'np':
            x_y_lam = x_y_lam.tolist()
        return x_y_lam

    '''
    #WIP
    def simulate_events(self, start_time, max_time):
        pass

    def ESTProcess(mu, k0, w, T):
        p = pois(mu*T)
        times = np.random.uniform(0, T, p) #(low, high, size)
        counts = 0
        countf = p-1

        while(countf>=counts):
            p=pois(k0)    #each event generates p offspring according to a Poisson r.v. with parameter k0
            for j in range(0, p):
                temp=times[counts]-np.log(np.random.rand())/w    
                if(temp < T):    
                    times = np.append(times, temp)
                    countf=countf+1
            counts=counts+1

        times = times[0:countf]      
        return times

    def pois(self, S):
        if S <= 100:
            temp = -1*S
            L = exp(temp)
            k = 0
            p = 1
            while p > L:
                k = k + 1
                p = p * random()
            p = k - 1
        else:
            p = floor(S + sqrt(S) * random())
        return int(p)
    '''    

