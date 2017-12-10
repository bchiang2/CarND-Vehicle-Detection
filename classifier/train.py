import glob
import numpy as np
import time
import os
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from classifier.hog_classify import extract_image_features as hog_extract_features
from classifier.color_classify import extract_image_features as color_extract_features
from classifier.helpers import open_image_file
from sklearn.model_selection import train_test_split

TEST_SIZE = None
SVC = None
SCALER = None
PICKLED_FILE_PATH = r"svc_model.pickle"


def load_image_paths():
    # Read in car and non-car images
    images = glob.glob('classifier/data/*/*/*.png')
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(open_image_file(image))
        else:
            cars.append(open_image_file(image))
    return cars, notcars


def extract_features(image):
    hog_features = hog_extract_features(image)
    color_features = color_extract_features(image)
    return np.concatenate((hog_features, color_features), axis=0)


def fit_model():
    cars, notcars = load_image_paths()
    if TEST_SIZE:
        cars, notcars = cars[:TEST_SIZE], notcars[:TEST_SIZE]

    car_features = [extract_features(car) for car in cars]
    notcar_features = [extract_features(notcar) for notcar in notcars]

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    return svc, X_scaler


def load_model(rebuild=False):
    global SVC, SCALER
    if os.path.exists(PICKLED_FILE_PATH) and not rebuild:
        print('Using pickled file')
        with open(PICKLED_FILE_PATH, 'rb') as f:
            data_set = pickle.load(f)
            SVC = data_set['svc']
            SCALER = data_set['scaler']
    else:
        print('Making pickled file')
        svc, scaler = fit_model()
        data_set = {'svc': svc, 'scaler': scaler}
        with open(PICKLED_FILE_PATH, 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
        SVC = data_set['svc']
        SCALER = data_set['scaler']

load_model(rebuild=True)