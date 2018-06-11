from flask import Flask
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import pickle
import io
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#process data
def process_data():
    df = pd.read_csv('final.csv')
    filenames = df.IMAGES

    for i in range (len (df)):
        filename = 'C:/Users/rjhosler/Documents/MIT-CBCL-facerec-database/training-originals/training_used_in_fpiv04_paper/' + filenames[i]
        image = load_img (filename, target_size = (30, 30))
        numpy_image = img_to_array (image)
        numpy_image = np.reshape (numpy_image, (1, 30*30*3))
        if (i > 0):
            image_array = np.append (image_array, numpy_image, axis = 0)
        else:
            image_array = np.copy (numpy_image)
    return image_array


#create and compile a model
def create_model():
    #autoencoder model

    model = Sequential ()

    model.add (Dense (300, input_shape = (2700,)))
    model.add (Activation ('relu'))
    model.add (Dropout (0.2))

    model.add (Dense (100))
    model.add (Activation ('relu'))
    model.add (Dropout (0.2))

    model.add (Dense (20))
    model.add (Activation ('relu'))

    model.add (Dense (100))
    model.add (Activation ('relu'))
    model.add (Dropout (0.2))

    model.add (Dense (300))
    model.add (Activation ('relu'))
    model.add (Dropout (0.2))

    model.add (Dense (2700))
    model.compile (loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mean_squared_error'])

    return model

#train the model on the processed data
def train_model (image_array, model):
    #run through the auto encoder with some noise for it to filter out.
    for i in range (500):
        noise = np.random.normal (loc = 0.0, scale = .75, size = image_array.shape)
        noisy_image = image_array + noise
        model.fit (noisy_image, image_array, batch_size = 128, epochs = 1, verbose = 1)
    return model

#retreive data from the model
def retreive_data (image_array_cnn, model_cnn, train):
    #method to retrieve features from the auto encoder
    """
    model2 = Sequential ()

    model2.add (Dense (300, input_shape = (2700,), weights = model.layers [0].get_weights ()))
    model2.add (Activation ('relu'))
    model2.add (Dropout (0.2))

    model2.add (Dense (100, weights = model.layers [3].get_weights ()))
    model2.add (Activation ('relu'))
    model2.add (Dropout (0.2))

    model2.add (Dense (20, weights = model.layers[6].get_weights ()))
    new_features = model2.predict (image_array)
    data = new_features

    if (train == True):
        #append a 1 'parole' and a 0 'no parole' to parts of the data (first 14 and other 16)
        binary_parole = np.zeros ((30, 1))
        binary_parole [0:14,] = 1
        data = np.column_stack((new_features, binary_parole))

    return data
    """
    #retrieve features from cnn
    model2_cnn = Sequential ()

    #from layer 1
    model2_cnn.add (Conv2D (32, kernel_size = (3, 3),
    activation = 'relu',
    input_shape = (30, 30, 3), 
    weights = model_cnn.layers [0].get_weights ()))
    model2_cnn.add (MaxPooling2D (pool_size = (2, 2)))

    #from layer 2
    model2_cnn.add (Conv2D (64, (3, 3), weights = model_cnn.layers [3].get_weights (), activation = 'relu'))
    model2_cnn.add (MaxPooling2D (pool_size = (2, 2)))
    model2_cnn.add (Dropout (0.25))

    #from layer 3 (flatened out data)
    #model2_cnn.add (Flatten ())
    #model2_cnn.add (Dense (128, weights = model_cnn.layers [6].get_weights ()))

    new_features_cnn = model2_cnn.predict_classes (image_array_cnn)
    data = new_features_cnn
    return data

#predict with the model
def rf_model (data):
    #split train data to train and test
    X_train, X_test, y_train, y_test = train_test_split(data [:,0:20], data [:,20], test_size = 0.5)

    #predict 'parole' or 'not parole' with a random forest classifier
    clf = RandomForestClassifier (n_jobs = -1, n_estimators = 10000, 
                              min_samples_leaf = 3, max_depth = 4,
                              min_samples_split = 3)
    clf.fit (X_train, y_train)

    return clf
        
"""
app = Flask(__name__)

@app.route('/hello')
def hello_world():
    image_array = process_data()
    #model = create_model()
    #model = train_model (image_array, model)

    # serialize model to JSON
    #model_json = model.to_json()
    #with open("model.json", "w") as json_file:
    #    json_file.write(model_json)
        
    # serialize weights to HDF5
    #model.save_weights("model.h5")
    
    #print("Saved model to disk")

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("model.h5")
    
    data = retreive_data (image_array, loaded_model)
    prediction = predict (data)
    prediction = np.array2string(prediction)
    return prediction

if __name__ == '__main__':
    app.add_url_rule('/', 'hello', hello_world)
    app.run()
"""
from flask import Flask, redirect, url_for, request
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    # load json and create model
    #json_file = open('model.json', 'r')
    #json_file = open('model_2.json', 'r')
    json_file = open('model_cnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights("model_cnn.h5")

    filename = 'C:/Users/rjhosler/Documents/test_images/' + name

    image = load_img (filename, target_size = (30, 30))
    numpy_image = img_to_array (image)
    numpy_image = numpy_image.astype('float32')
    numpy_image /= 255
    numpy_image = numpy_image.reshape (-1, 30, 30, 3)
    
    input_data = retreive_data (numpy_image, loaded_model, False)
    input_data = input_data.reshape (1, 36)

    #filename = 'rf_model.sav'
    #filename = 'rf_model2.sav'
    filename = 'rf_model4.sav'

    # load the model from disk
    loaded_clf = pickle.load(open(filename, 'rb'))

    prediction = loaded_clf.predict (input_data)  
    
    if (prediction > 0.5):
        prediction = 'You are a parolee'
    else:
        prediction = 'You are not a parolee'
    
    return prediction

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

if __name__ == '__main__':
   app.run()
