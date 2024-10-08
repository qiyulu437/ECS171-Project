import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  

import tensorflow as tf

import skimage as ski

from sklearn.utils.class_weight import compute_class_weight as ccw

import keras
from keras.metrics import F1Score as f1
from keras import layers, models
from keras.utils import image_dataset_from_directory as idft
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences as pad
from keras.callbacks import EarlyStopping as ES

import keras_tuner as kert

from scikeras.wrappers import KerasClassifier as KC

from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import StratifiedKFold as SKF

import numpy as np

gpus = tf.config.list_physical_devices('GPU')
print(f"Available gpus: {len(gpus)}")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            '''tf.config.experimental.set_virtual_device_configuration(gpu,
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])'''
    except RuntimeError as e:
        print(e)
        
tf.config.optimizer.set_jit("autoclustering")
keras.mixed_precision.set_global_policy(keras.mixed_precision.Policy("mixed_float16"))


# 35 nodes and over appear to cause the model to overfit
# Best learning rate is around 0.01, but definitely needs further tuning because of model instability.

nodes_init = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
kernel_size_1 = [1, 3, 5, 7]
hidden_fcn = ["leaky_relu", "relu", "elu", "silu"]
pool_size_1 = [2, 3]

nodes_mid_1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
nodes_mid_2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
kernel_size_2 = [1, 3, 5, 7]
nodes_fin_1 = [32, 64, 128, 256]
nodes_fin_2 = [32, 64, 128, 256]
kernel_size_3 = [1, 3, 5, 7]

pool_size_2 = [2, 3]
nodes_dense_1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
nodes_dense_2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
output_fcn = ["sigmoid"]

lr = [0.00001, 0.0001, 0.001, 0.01]
pool_size_3 = [2, 3]

weights = {0: 0.554, 1: 5.113}

f1_score = f1()


class ClearMemory(keras.callbacks.Callback):
    def on_trial_end(self, logs = None):
        keras.backend.clear_session()


def build_model_final(num_nodes, kern, pool, learn):
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
    # Generate a tf.data.Dataset object from the 256 folder
    # Binary labeling of either waldo or not waldo
    
    CNN = models.Sequential(name = "Waldo")
    CNN.add(layers.Input(shape = (128, 128, 3)))
    CNN.add(layers.Conv2D(num_nodes, (kern, kern), activation = "leaky_relu", padding = "same"))
    CNN.add(layers.MaxPooling2D((pool, pool)))
    CNN.add(layers.Conv2D(num_nodes * 2, (kern, kern), activation = "leaky_relu", padding = "same"))
    CNN.add(layers.MaxPooling2D((pool, pool)))
    #CNN.add(layers.Conv2D(num_nodes * 3, (kern, kern), activation = "leaky_relu", padding = "same"))

    CNN.add(layers.Flatten())
    #CNN.add(layers.Dense(hp.Choice("nodes", nodes) * 2, activation = "leaky_relu"))
    #CNN.add(layers.Dense(num_nodes * 2, activation = "leaky_relu"))
    CNN.add(layers.Dense(1, activation = "sigmoid"))
    adam = Adam(learning_rate = learn)

    CNN.compile(optimizer = adam, 
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ["accuracy", "precision", "recall", f1_score])
    
    return CNN

def build_model(hp):
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
    # Generate a tf.data.Dataset object from the 256 folder
    # Binary labeling of either waldo or not waldo    
    CNN = models.Sequential(name = "Waldo")
    CNN.add(layers.Input(shape = (256, 256, 3)))
    
    # Starting layer
    CNN.add(layers.Conv2D(hp.Choice("nodes_init", nodes_init), 
                          (hp.Choice("kernel_size_1", kernel_size_1), hp.Choice("kernel_size_1", kernel_size_1)), 
                          activation = hp.Choice("hidden_activation_function", hidden_fcn),
                          padding = "same"))
    
    CNN.add(layers.MaxPooling2D((hp.Choice("pool_size_1", pool_size_1), hp.Choice("pool_size_1", pool_size_1))))

    # Middle layer
    CNN.add(layers.Conv2D(hp.Choice("nodes_mid_1", nodes_mid_1), 
                          (hp.Choice("kernel_size_2", kernel_size_2), hp.Choice("kernel_size_2", kernel_size_2)), 
                          activation = hp.Choice("hidden_activation_function", hidden_fcn),
                          padding = "same"))
    CNN.add(layers.MaxPooling2D((hp.Choice("pool_size_2", pool_size_2), hp.Choice("pool_size_2", pool_size_2))))
    CNN.add(layers.Conv2D(hp.Choice("nodes_mid_2", nodes_mid_2), 
                          (hp.Choice("kernel_size_2", kernel_size_2), hp.Choice("kernel_size_2", kernel_size_2)), 
                          activation = hp.Choice("hidden_activation_function", hidden_fcn), 
                          padding = "same"))
    CNN.add(layers.MaxPooling2D((hp.Choice("pool_size_3", pool_size_3), hp.Choice("pool_size_3", pool_size_3))))
    

    # Deep layer
    CNN.add(layers.Conv2D(hp.Choice("nodes_fin_1", nodes_fin_1), 
                          (hp.Choice("kernel_size_3", kernel_size_3), hp.Choice("kernel_size_3", kernel_size_3)), 
                          activation = hp.Choice("hidden_activation_function", hidden_fcn),
                          padding = "same"))
    CNN.add(layers.Conv2D(hp.Choice("nodes_fin_2", nodes_fin_2), 
                          (hp.Choice("kernel_size_3", kernel_size_3), hp.Choice("kernel_size_3", kernel_size_3)), 
                          activation = hp.Choice("hidden_activation_function", hidden_fcn),
                          padding = "same"))

    CNN.add(layers.GlobalAveragePooling2D())
    # Dense layer
    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(hp.Choice("nodes_dense_1", nodes_dense_1), activation = hp.Choice("hidden_activation_function", hidden_fcn)))
    CNN.add(layers.Dense(hp.Choice("nodes_dense_2", nodes_dense_2), activation = hp.Choice("hidden_activation_function", hidden_fcn)))
    CNN.add(layers.Dense(1, activation = hp.Choice("output_activation_function", output_fcn)))
    adam = Adam(learning_rate = hp.Choice("learning_rate", lr))

    CNN.compile(optimizer = adam, 
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ["accuracy", "precision", "recall", f1_score],
                jit_compile = True)
    
    return CNN

def tune_hyperparameters(training, testing, trials):
    tuner = kert.GridSearch(hypermodel = build_model,
                              objective = kert.Objective("val_f1_score", direction = "max"),
                              max_trials = trials, 
                              seed = 123,
                              project_name = "CNN Tuning")

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
    
    early_stop = ES(monitor = "val_f1_score", patience = 10, restore_best_weights = True)
            
    tuner.search(x = training, 
                 epochs = 50, 
                 validation_data = testing,
                 class_weight = weights,
                 callbacks = [early_stop, ClearMemory()])
    
    tuner.results_summary(num_trials = 30)
    '''best_model = tuner.get_best_models()[0]
    best_params = tuner.get_best_hyperparameters(num_trials = 1)[0]
    for key, value in best_params.values.items():
        print(f"{key}: {value}")'''



waldo_images = r"256"

# Focusing on optimizng learning process and improving stability.


training, testing = idft(waldo_images, 
                         validation_split = 0.3, 
                         subset = "both", 
                         label_mode = "binary", 
                         seed = 123, 
                         labels = "inferred", 
                         image_size = (256, 256), 
                         batch_size = 32)



tune_hyperparameters(training, testing, trials = 100)
CNN = build_model_final(21, 
                        3, 
                        7, 
                        0.012)

