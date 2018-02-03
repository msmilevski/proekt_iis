from skimage import io
import cv2
import csv

url = "https://images-na.ssl-images-amazon.com/images/I/91gnU6Qm7LL._UL1500_.jpg"
file_path = 'data/outfit_products.csv'
directory_path = 'data/images/'

def url_to_image(url, name, directory_path):
    image = io.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (299,299), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(directory_path+name, image)
    #cv2.imshow("slika", image)
    #cv2.waitKey(0)

    return image


with open(file_path, newline='', encoding="utf8") as product_outfits:
    reader = csv.reader(product_outfits, delimiter=',', quotechar="|")
    i = 0
    for row in reader:
        if i != 0:
            name = str(row[0])+".jpg"
            url = row[-1].replace('"', '')
            url_to_image(url, name, directory_path)
            break
        i += 1
