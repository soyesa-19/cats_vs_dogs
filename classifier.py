from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import rmtree
import cv2

path = "./dataSet/datasets/catsvsdogs/images/"

file_names = [i for i in listdir(path) if isfile(join(path, i))]

# print(str(len(file_names)) + " images")
dog_count = 0
cat_count = 0
size = 150
training_size = 1000
test_size = 500
training_images = []
training_labels = []
test_images = []
test_labels = []

dog_train = "./dataSet/datasets/catsvsdogs/train/train_dogs/"
cat_train = "./dataSet/datasets/catsvsdogs/train/train_cats/"
dog_test = "./dataSet/datasets/catsvsdogs/test/test_dogs/"
cat_test = "./dataSet/datasets/catsvsdogs/test/test_cats/"

def make_dir(directories):
    if exists(directories):
        rmtree(directories)
    makedirs(directories)

make_dir(dog_train)
make_dir(dog_test)
make_dir(cat_train)
make_dir(cat_test)

def get_zeros(number):
    if number>10 and number<100:
        return "0"
    elif number<10:
        return "00"
    else:
        return ""

for i, file in enumerate(file_names):
    if file_names[i][0] == "d":
        dog_count += 1
        img = cv2.imread(path+file)
        img = cv2.resize(img, (size, size), interpolation= cv2.INTER_AREA)
        if dog_count<= training_size:
            training_images.append(img)
            training_labels.append(1)
            zeros = get_zeros(dog_count)
            cv2.imwrite(dog_train + "dog" + str(zeros) + str(dog_count) + ".png", img)
        elif dog_count>1000 and dog_count <= (training_size+test_size):
            test_images.append(img)
            test_labels.append(1)
            zeros = get_zeros(dog_count)
            cv2.imwrite(dog_test + "dog" + str(zeros) + str(dog_count-1000) + ".png", img)
        
    if file_names[i][0] == "c":
        cat_count += 1
        img = cv2.imread(path+file)
        img = cv2.resize(img, (size, size), interpolation= cv2.INTER_AREA)
        if cat_count <= training_size:
            training_images.append(img)
            training_labels.append(0)
            zeros = get_zeros(cat_count)
            cv2.imwrite(cat_train + "cat" + str(zeros) + str(cat_count) + ".png", img)
        
        elif cat_count>1000 and cat_count <= (training_size + test_size):
            test_images.append(img)
            test_labels.append(0)
            zeros = get_zeros(cat_count)
            cv2.imwrite(cat_test + "cat" + str(zeros) + str(cat_count-1000) + ".png", img)

    if dog_count == training_size+test_size and cat_count == training_size+test_size:
        break

print("training and test data extraction completed")



