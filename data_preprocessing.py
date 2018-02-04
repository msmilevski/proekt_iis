from skimage import io
import cv2
import csv
import json
import operator


# Helper function for reading images from url, resize them and saving them to the given directory
def url_to_image(url, name, directory_path):
    image = io.imread(url)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(directory_path + name, image)

    # cv2.imshow("slika", image)


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

    sorted_dictironary = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_dictironary)
    with open('clothes_vocab.json', 'w') as outfile:
        json.dump(sorted_dictironary, outfile)


create_clothes_vocabulary()
