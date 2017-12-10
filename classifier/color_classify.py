import matplotlib.image as mpimg
import numpy as np
import cv2


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_image_features(image, spatial_size=(32, 32),
                           hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
    # Append the new feature vector to the features list
    return np.concatenate((spatial_features, hist_features), axis=0)
