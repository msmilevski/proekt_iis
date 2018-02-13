from sklearn import linear_model
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.externals import joblib
import h5py


def train_data(train_x, train_y, valid_x, valid_y):
    num_train_samples = len(train_x)
    num_similarity = len(train_y)
    clf = PassiveAggressiveRegressor()
    start_x = 0
    start_y = 0
    print("PassiveAggressiveRegressor")
    while start_x < num_train_samples and start_y < num_similarity:
        train_slice_x = train_x[start_x: start_x + 1000]
        train_slice_y = train_y[start_y: start_y + 500]
        start_x += 1000
        start_y += 500
        summed_train_x = []

        for j in range(0, len(train_slice_x), 2):
            summed_train_x.append(train_slice_x[j] + train_slice_x[j + 1])

        summed_train_x = np.array(summed_train_x)
        clf.partial_fit(summed_train_x, train_slice_y)

    num_valid_samples = len(valid_x)
    num_similarity = len(valid_y)
    summed_valid_y = []
    for i in range(0, num_valid_samples, 2):
        summed_valid_y.append(valid_x[i]+valid_x[i+1])

    summed_valid_y = np.array(summed_valid_y)

    print("coefficient of determination R^2 of the predictiot: ")
    print(clf.score(summed_valid_y, valid_y))
    #joblib.dump(clf, 'sgdregressor.pkl')


training_file = h5py.File("data/train_image_embeddings.hdf5", 'r')
valid_file = h5py.File("data/valid_image_embeddings.hdf5", 'r')
image_embeddings = training_file["image_embeddings"]
print(len(image_embeddings))
similarity = training_file["similarity"]
print(len(valid_file["image_embeddings"]))


train_data(image_embeddings, similarity, valid_file["image_embeddings"], valid_file["similarity"])
