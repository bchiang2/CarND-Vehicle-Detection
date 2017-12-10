import glob
import numpy as np
import os
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from classifier.features.hog_classify import extract_image_features as hog_extract_features
from classifier.features.color_classify import extract_image_features as color_extract_features
from classifier.helpers import open_image_file
from sklearn.model_selection import train_test_split
import config

TEST_SIZE = None
SVC = None
SCALER = None
PICKLED_FILE_PATH = r"svc_model.pickle"
IMAGE_PICKLED_FILE_PATH = r"image.pickle"


def load_images():
    if not config.RELOAD_IMAGE and os.path.exists(IMAGE_PICKLED_FILE_PATH):
        with open(IMAGE_PICKLED_FILE_PATH, 'rb') as f:
            data_set = pickle.load(f)
            cars = data_set['cars']
            notcars = data_set['notcars']
    else:
        # Read in car and non-car images
        types = ['classifier/data/*/*/*.png', 'classifier/data/*/*/*.jpg']
        images = []
        for files in types:
            images.extend(glob.glob(files))
        cars = []
        notcars = []
        for image in images:
            standard_rgb_image = open_image_file(image)
            if len(standard_rgb_image.shape) != 3:
                print("Issue with {}".format(image))
                continue
            if 'non-vehicles' in image:
                notcars.append(standard_rgb_image)
            else:
                cars.append(standard_rgb_image)

        data_set = {'cars': cars, 'notcars': notcars}
        with open(IMAGE_PICKLED_FILE_PATH, 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
    return cars, notcars


def extract_features(image):
    color_features = color_extract_features(image)
    hog_features = hog_extract_features(image)
    return np.concatenate((hog_features, color_features), axis=0)


def fit_model():
    cars, not_cars = load_images()
    if TEST_SIZE:
        cars, not_cars = cars[:TEST_SIZE], not_cars[:TEST_SIZE]

    car_features = [extract_features(car) for car in cars]
    notcar_features = [extract_features(notcar) for notcar in not_cars]

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
        scaled_X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=rand_state
    )

    # Use a linear SVC
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    svc.fit(X_train, y_train)


    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    n_predict = 100
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])

    return svc, X_scaler


def load_model(rebuild):
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


load_model(rebuild=config.REBUILD)
