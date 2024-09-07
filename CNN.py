import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

import skimage as ski

import keras
from keras.metrics import F1Score as f1
from keras import layers, models
from keras.utils import image_dataset_from_directory as idft
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences as pad

import keras_tuner as kert

from scikeras.wrappers import KerasClassifier as KC

from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import StratifiedKFold as SKF

import numpy as np



nodes = [16, 32, 64]
kernel_size = [3, 4, 5, 6, 7, 8]
pool_size = [3, 4, 5, 6, 7, 8]
lr = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]

f1_score = f1()

def build_model(hp):
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
    # Generate a tf.data.Dataset object from the 256 folder
    # Binary labeling of either waldo or not waldo
    
    CNN = models.Sequential(name = "Waldo")
    CNN.add(layers.Input(shape = (256, 256, 3)))
    CNN.add(layers.Conv2D(hp.Choice("nodes", nodes), (hp.Choice("kernel_size", kernel_size), hp.Choice("kernel_size", kernel_size)), activation = "sigmoid", padding = "same", strides = 1))
    CNN.add(layers.MaxPooling2D((hp.Choice("pool_size", pool_size), hp.Choice("pool_size", pool_size))))
    CNN.add(layers.Conv2D(hp.Choice("nodes", nodes) * 2, (hp.Choice("kernel_size", kernel_size), hp.Choice("kernel_size", kernel_size)), activation = "sigmoid", padding = "same", strides = 1))
    CNN.add(layers.MaxPooling2D((hp.Choice("pool_size", pool_size), hp.Choice("pool_size", pool_size))))
    CNN.add(layers.Conv2D(hp.Choice("nodes", nodes) * 4, (hp.Choice("kernel_size", kernel_size), hp.Choice("kernel_size", kernel_size)), activation = "sigmoid", padding = "same", strides = 1))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(hp.Choice("nodes", nodes) * 4, activation = "sigmoid"))
    CNN.add(layers.Dense(hp.Choice("nodes", nodes) * 2, activation = "sigmoid"))
    CNN.add(layers.Dense(1, activation = "sigmoid"))
    adam = Adam(learning_rate = hp.Choice("learning rate", lr))

    CNN.compile(optimizer = adam, 
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ["accuracy", "precision", "recall", f1_score])
    
    return CNN

def tune_hyperparameters(training, testing, cv):
    #build_model(kert.HyperParameters())
    tuner = kert.RandomSearch(build_model,
                              objective = kert.Objective("val_f1_score", direction = "max"),
                              max_trials = cv, 
                              seed = 123)

    x = []
    y = []

    for features, labels in training:
        x.append(features.numpy())
        y.append(labels.numpy())
        
    x_train = np.concatenate(x, axis = 0)
    y_train = np.concatenate(y, axis = 0)
    
    x = []
    y = []

    for features, labels in testing:
        x.append(features.numpy())
        y.append(labels.numpy())
        
    x_test = np.concatenate(x, axis = 0)
    y_test = np.concatenate(y, axis = 0)
            
    tuner.search(x_train, y_train, epochs = 16, validation_data = [x_test, y_test])
    best = tuner.get_best_models()[0]
    best



waldo_images = r"256"
original_images = r"original-images"

keras.backend.clear_session()

training, testing = idft(waldo_images, 
                         validation_split = 0.2, 
                         subset = "both", 
                         label_mode = "binary", 
                         seed = 123, 
                         labels = "inferred", 
                         image_size = (256, 256), 
                         batch_size = 5)


tune_hyperparameters(training, testing, cv = 5)







