import glob
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from classifier.hog_classify import extract_features as hog_extract_features
from classifier.color_classify import extract_features as color_extract_features
from sklearn.model_selection import train_test_split

TEST_SIZE = None

def load_image_paths():
    # Read in car and non-car images
    images = glob.glob('data/*/*/*.png')
    cars = []
    notcars = []
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)
    return cars, notcars

def concat_features(image_paths):
    hog_features = hog_extract_features(image_paths)
    color_features = color_extract_features(image_paths)
    return np.concatenate((hog_features, color_features), axis=1)

def extract_features():
    cars, notcars = load_image_paths()
    if TEST_SIZE:
        cars, notcars = cars[:TEST_SIZE], notcars[:TEST_SIZE]

    car_features = concat_features(cars)
    notcar_features = concat_features(notcars)


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
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')




print(extract_features())