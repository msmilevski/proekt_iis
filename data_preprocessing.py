from skimage import io
import cv2
import csv
import json
import operator
import itertools
from glob import glob
import os
import numpy as np
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import time
import h5py


# Helper function for reading images from url, resize them and saving them to the given directory
def url_to_image(url, name, directory_path):
    image = io.imread(url)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(directory_path + name, image)


# Read images from csv file and save them in a directory
def save_images_from_file(file_path='data/outfit_products.csv', directory_path='./data/images/'):
    with open(file_path, newline='', encoding="utf8") as product_outfits:
        reader = csv.reader(product_outfits, delimiter=',', quotechar="|")
        i = 0
        for row in reader:
            if i != 0:
                name = str(row[0]) + ".jpg"
                url = row[-1].replace('"', '')
                print(name)
                url_to_image(url, name, directory_path)
            i += 1


# Finds all the corrupted images
def find_index_of_broken_images(file_path='data/outfit_products.csv'):
    with open(file_path, newline='', encoding="utf8") as product_outfits:
        reader = csv.reader(product_outfits, delimiter=',', quotechar="|")
        i = 0
        indices = []
        for row in reader:
            if i != 0:
                url = row[-1].replace('"', '')
                index = row[0]
                if '.gif' in url:
                    indices.append(index)
            i += 1

    print(indices)


# Function for creating a vocabulary of clothes' names
def create_clothes_vocabulary(file_path='data/outfit_products.csv'):
    with open(file_path, newline='', encoding="utf8") as product_outfits:
        reader = csv.reader(product_outfits, delimiter=',', quotechar="|")
        i = 0
        dictionary = {}
        for row in reader:
            if i != 0:
                labels = row[1].split()
                for j in range(len(labels)):
                    word = labels[j].lower()
                    if word not in dictionary.keys():
                        dictionary[word] = 1
                    else:
                        dictionary[word] += 1
            i += 1

    sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_dictionary)
    with open('clothes_vocab.json', 'w') as outfile:
        json.dump(sorted_dictionary, outfile)


# Function for creating positive and negative training samples
def create_training_pairs(file_path='./data/products_outfits.csv', image_directory='./data/images/',
                          num_negative_samples=4):
    with open(file_path, newline='', encoding="utf8") as product_outfits:
        reader = csv.reader(product_outfits, delimiter=',', quotechar="|")
        i = 0
        dictionary = {}
        positive_samples = []
        for row in reader:
            if i != 0:
                product_id = row[0].replace('"', '') + ".jpg"
                outfit_id = row[1].replace('"', '')

                if not (outfit_id in dictionary.keys()):
                    dictionary[outfit_id] = [product_id]
                else:
                    temp_list = dictionary[outfit_id]
                    temp_list.append(product_id)
                    dictionary[outfit_id] = temp_list

                if not (product_id in positive_samples):
                    positive_samples.append(product_id)
            i += 1

    outfit_ids = dictionary.keys()
    positive_sample_images = []
    # Creating positive samples
    for idx in outfit_ids:
        temp_list = dictionary[idx]
        temp_combinations = list(itertools.combinations(temp_list, 2))
        print(temp_combinations)
        for positive_comb in temp_combinations:
            positive_comb = [list(positive_comb), 1]
            positive_sample_images.append(positive_comb)

    # Getting all image file names
    image_path_names = [y for x in os.walk(image_directory) for y in glob(os.path.join(x[0], "*.jpg"))]
    all_image_names = []
    for path in image_path_names:
        image_name = path.split("\\")[-1]
        all_image_names.append(image_name)

    length_all_images = len(all_image_names)
    # Creating negative samples
    negative_sample_images = []
    for sample in positive_samples:
        k = 0
        while k < num_negative_samples:
            rand_idx = np.random.randint(0, length_all_images)
            rand_img_name = all_image_names[rand_idx]
            if sample != rand_img_name:
                negative_sample_images.append([[sample, rand_img_name], 0])
                k += 1

    # Concatenating the positive and negative samples
    # And also shuffling them
    finished_dataset = positive_sample_images + negative_sample_images
    np.random.shuffle(finished_dataset)
    finished_dataset = np.array(finished_dataset)

    # Splitting data to train, valid and test data
    train_barrier = int(finished_dataset.shape[0] * 0.8)
    valid_barrier = int(finished_dataset.shape[0] * 0.1)
    test_barrier = int(finished_dataset.shape[0] * 0.1)
    train_dataset = finished_dataset[0:train_barrier]
    valid_dataset = finished_dataset[train_barrier:train_barrier + valid_barrier]
    test_dataset = finished_dataset[train_barrier + valid_barrier:train_barrier + valid_barrier + test_barrier]

    # Saving data to file so we don't have to do this all the time
    np.save("train-data", train_dataset)
    np.save("valid-data", valid_dataset)
    np.save("test-data", test_dataset)

