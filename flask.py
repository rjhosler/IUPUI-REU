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
    def __init__ (self, filename):
        df = pd.read_csv (filename)
        data = np.stack((df.XCOORD, df.YCOORD), axis = 1)
        data = data [0:1000, :]
        self._data = data

        X1 = (np.amin((data[:,0])) - np.amax((data[:,0]))) * np.random.random_sample((30,1)) + np.amax((data[:,0]))
        X2 = (np.amin((data[:,1])) - np.amax((data[:,1]))) * np.random.random_sample((30,1)) + np.amax((data[:,1]))
        X = np.stack ((X1, X2), axis = 1)
        self._centers  = X.reshape ((30,2))       

    #method to implement the wasserstein algorithm
    def wasserstein (self):
        theta = 0.5
        a = len (self._data)
        b = len (self._centers)
        X = self._centers
        data = self._data
        lam = 100
        while (1):
            M = cdist (X, data, 'euclidean')
            K = math.e ** (-1 * lam * M)
            Kt = np.divide (K, a)
            u = np.transpose (np.ones(len(X)) / len(X))
            change = 1
            while (change > 0.0001):
                oldu = u
                p = np.matmul (np.transpose (K), u)
                f = np.divide (b, p)
                g = np.matmul (Kt, f)
                u = np.transpose (np.ones(len(X)) / g)
                change = la.norm (u - oldu)
            V = np.divide (b, np.matmul (np.transpose(K), u))        
            T = np.matmul (np.matmul (np.diag(u), K), np.diag (V))
            oldX = X
            X = (X * (1 - theta)) + (np.divide (np.matmul (T, data), a) * theta)
            print (la.norm (oldX - X))
            if (la.norm (oldX - X) < 0.005):
                return X

    def process_data (self):
        self._centers = self.wasserstein()

    def get_data (self):
        return self._data

    def get_centers (self):
        return self._centers

from flask import Flask, redirect, url_for, request, render_template_string        
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    cluster = Cluster (name)
    cluster.process_data()
    data = cluster.get_data()
    centers = cluster.get_centers()
    plt.scatter(data[:,0], data[:,1]),
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 100, alpha=0.5)

    return "success"

@app.route('/login', methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form.get('nm', None)
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

if __name__ == '__main__':
   app.run()
