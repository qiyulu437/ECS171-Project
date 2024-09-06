import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

import skimage as ski

import keras
from keras import layers, models
from keras.utils import image_dataset_from_directory as idft
from keras.optimizers import Adam

from scikeras.wrappers import KerasClassifier as KC

from sklearn.model_selection import GridSearchCV as GSCV

import numpy as np



def build_model(lr, num_nodes, pool_size, kernel_size):
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
    # Generate a tf.data.Dataset object from the 256 folder
    # Binary labeling of either waldo or not waldo
    
    CNN = models.Sequential(name = "Waldo")
    CNN.add(layers.Input(shape = (256, 256, 3)))
    CNN.add(layers.Conv2D(num_nodes, (kernel_size, kernel_size), activation = "sigmoid"))
    CNN.add(layers.MaxPooling2D((pool_size, pool_size)))
    CNN.add(layers.Conv2D(num_nodes, (kernel_size, kernel_size), activation = "sigmoid"))
    CNN.add(layers.MaxPooling2D((pool_size, pool_size)))
    CNN.add(layers.Conv2D(num_nodes, (kernel_size, kernel_size), activation = "sigmoid"))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(num_nodes, activation = "sigmoid"))
    CNN.add(layers.Dense(num_nodes, activation = "sigmoid"))
    CNN.add(layers.Dense(1, activation = "sigmoid"))
    adam = Adam(learning_rate = lr)

    CNN.compile(optimizer = adam, 
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ["accuracy", "precision", "recall", "f1_score"])
    
    return CNN

def tune_hyperparameters(CNN, param_grid, training, n_jobs, cv):
    grid = GSCV(estimator = CNN, param_grid = param_grid, n_jobs = n_jobs, cv = cv, scoring = ["accuracy", "precision", "recall", "f1"], refit = False)

    x = []
    y = []

    for features, labels in training:
        x.append(features.numpy())
        y.append(labels.numpy())
        
    x = np.concatenate(x, axis = 0)
    y = np.concatenate(y, axis = 0)
            
    grid_result = grid.fit(x, y, class_weight = {0: 1.0, 1: 8.0})

    optimal_params = grid_result.best_params_
    print(f"The best settings are {optimal_params}")

    text_file = os.open("optimal.txt", os.O_RDWR)
    os.write(text_file, str.encode(optimal_params))
    os.close(text_file)
    
param_grid = {
    "lr": [0.00001, 0.0001],
    "epochs": [8, 16],
    "num_nodes": [16, 32],
    "pool_size": [2, 3], # Min 2, max 8
    "kernel_size": [3, 4] # Min 3, max 7
}



waldo_images = r"256"
original_images = r"original-images"

training, testing = idft(waldo_images, 
                         validation_split = 0.2, 
                         subset = "both", 
                         label_mode = "binary", 
                         seed = 123, 
                         labels = "inferred", 
                         image_size = (256, 256), 
                         batch_size = 32)

CNN = KC(model = build_model, 
         epochs = 50, 
         batch_size = 32, 
         lr = 0.00001, 
         num_nodes = 64, 
         pool_size = 2, 
         kernel_size = 3)

tune_hyperparameters(CNN, param_grid, training, n_jobs = 2, cv = 5)






