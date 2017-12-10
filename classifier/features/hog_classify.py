import numpy as np
from skimage.feature import hog
from classifier.helpers import normalize_array
import config


def get_hog_features(img, feature_vec=True):
    features = hog(
        img,
        orientations=config.HOG_ORIENTATION,
        pixels_per_cell=(config.PIXEL_PER_CELL_BLOCK, config.PIXEL_PER_CELL_BLOCK),
        cells_per_block=(config.CELL_PER_BLOCK, config.CELL_PER_BLOCK),
        transform_sqrt=True,
        feature_vector=feature_vec
    )
    return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_image_features(image,
                           orient=9,
                           pix_per_cell=16,
                           cell_per_block=4,
                           hog_channel='ALL'
                           ):
    # Create a list to append feature vectors to
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(get_hog_features(image[:, :, channel],
                                                 feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(image[:, :, hog_channel]
                                        , feature_vec=True)
    return hog_features
