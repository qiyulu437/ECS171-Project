import numpy as np

import skimage as ski

from sklearn.decomposition import PCA

import re

import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots as splt



# For initializing the row and col numbers of each subplot respective to the number of features extracted from the image
def nearest_factor(target):
    factor1 = np.sqrt(target).astype(int)
    factor2 = factor1

    while (factor1 * factor2) != target:
        if factor1 * factor2 > target:
            factor1 = factor1 - 1
            factor2 = factor1
        else:
            factor2 = factor2 + 1

    return factor1, factor2

def generate_plot(nrows: int, ncols: int, plot_labels_font: int, feature, feature_name: str, path: str, file_name: str, file_name_type: str):
    fig, ax = splt(nrows, ncols, figsize = (28, 11))
    fig.suptitle(feature_name + " graph", fontsize = plot_labels_font)
    fig.supxlabel("Horizontal pixel index", fontsize = plot_labels_font)
    fig.supylabel("Vertical pixel index", fontsize = plot_labels_font)
    # Display pixel intensity graph
    for i in range(feature.shape[2]):
        row = i // ncols
        col = i % ncols
        cax = ax[row, col].imshow(feature[:, :, i])
        ax[row, col].set_title(f"Feature {i}")
        cbar = plt.colorbar(cax, ax = ax[row, col], label = feature_name)
    
    fig.savefig(fname = os.path.join(path, file_name + file_name_type))
    plt.close()
    
def variance_graph(total_variance, file_name_index, title):
    plt.plot((total_variance / file_name_index) * 100)
    plt.title(title + " Explained Variance as a Function of the Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance %")
    plt.show()
    plt.close()



# Extract different resolution and color categories from image dataset. Total is 9 + the original images.
waldo_images = r"256\waldo\*.jpg"
not_waldo_images = r"256\notwaldo\*.jpg"

# 256 color images
waldo_dataset = ski.io.imread_collection(waldo_images)
not_waldo_dataset = ski.io.imread_collection(not_waldo_images)

file_name_index = 0

total_variance_pi = 0
total_variance_edges = 0
total_variance_texture = 0

# Traverse over every image in the 256 waldo folder
for img in waldo_dataset:
    pca_pi = PCA(n_components = 60)
    pca_edges = PCA(n_components = 120)
    pca_texture = PCA(n_components = 120)
    # Features' array shape is a tuple of (256, 256, n), where n is the number of features extracted.
    # Organized features extracted
    pixel_intensity = ski.feature.multiscale_basic_features(img, channel_axis = 2, edges = False, texture = False)
    edges = ski.feature.multiscale_basic_features(img, channel_axis = 2, intensity = False, texture = False)
    texture = ski.feature.multiscale_basic_features(img, channel_axis = 2, edges = False, intensity = False)
    
    pca_pi.fit(pixel_intensity[:, :, 0])
    pca_edges.fit(edges[:, :, 0])
    pca_texture.fit(texture[:, :, 0])
    
    total_variance_pi += np.cumsum(pca_pi.explained_variance_ratio_)
    total_variance_edges += np.cumsum(pca_edges.explained_variance_ratio_)
    total_variance_texture += np.cumsum(pca_texture.explained_variance_ratio_)

    # Directory
    path = "plots\\waldo"

    # Extract file name alone from collection
    file_name = waldo_dataset.files[file_name_index].replace("256\\waldo\\", "")
    file_name = re.sub(".jpg", "", file_name)
    
    nrows, ncols = nearest_factor(pixel_intensity.shape[2])
    generate_plot(nrows, ncols, 30, pixel_intensity, "Pixel intensity", path, file_name, "_pixel_intensity.jpg")
    
    nrows, ncols = nearest_factor(edges.shape[2])
    generate_plot(nrows, ncols, 30, edges, "Local gradient intensity", path, file_name, "_edges.jpg")
    
    nrows, ncols = nearest_factor(texture.shape[2])
    generate_plot(nrows, ncols, 30, texture, "Texture intensity", path, file_name, "_texture.jpg")
    
    file_name_index += 1
    
variance_graph(total_variance_pi, file_name_index, "Pixel Intensity")
variance_graph(total_variance_edges, file_name_index, "Edges")
variance_graph(total_variance_texture, file_name_index, "Texture")


file_name_index = 0

total_variance_pi = 0
total_variance_edges = 0
total_variance_texture = 0

# Traverse over every image in the 256 notwaldo 
for img in not_waldo_dataset:
    pca_pi = PCA(n_components = 60)
    pca_edges = PCA(n_components = 120)
    pca_texture = PCA(n_components = 120)
    # Features' array shape is a tuple of (256, 256, n), where n is the number of features extracted.
    # Organized features extracted
    pixel_intensity = ski.feature.multiscale_basic_features(img, channel_axis = 2, edges = False, texture = False)
    edges = ski.feature.multiscale_basic_features(img, channel_axis = 2, intensity = False, texture = False)
    texture = ski.feature.multiscale_basic_features(img, channel_axis = 2, edges = False, intensity = False)
    
    pca_pi.fit(pixel_intensity[:, :, 0])
    pca_edges.fit(edges[:, :, 0])
    pca_texture.fit(texture[:, :, 0])
    
    total_variance_pi += np.cumsum(pca_pi.explained_variance_ratio_)
    total_variance_edges += np.cumsum(pca_edges.explained_variance_ratio_)
    total_variance_texture += np.cumsum(pca_texture.explained_variance_ratio_)

    # Directory
    path = "plots\\notwaldo"

    # Extract file name alone from collection
    file_name = not_waldo_dataset.files[file_name_index].replace("256\\notwaldo\\", "")
    file_name = re.sub(".jpg", "", file_name)
    
    nrows, ncols = nearest_factor(pixel_intensity.shape[2])
    generate_plot(nrows, ncols, 30, pixel_intensity, "Pixel intensity", path, file_name, "_pixel_intensity.jpg")
    
    nrows, ncols = nearest_factor(edges.shape[2])
    generate_plot(nrows, ncols, 30, edges, "Local gradient", path, file_name, "_edges.jpg")
    
    nrows, ncols = nearest_factor(texture.shape[2])
    generate_plot(nrows, ncols, 30, texture, "Texture intensity", path, file_name, "_texture.jpg")
    
    file_name_index += 1
    
variance_graph(total_variance_pi, file_name_index, "Pixel Intensity")
variance_graph(total_variance_edges, file_name_index, "Edges")
variance_graph(total_variance_texture, file_name_index, "Texture")

print("DONE")

