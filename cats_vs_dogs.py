import cv2
import numpy
from keras.models import load_model

classifier = load_model("cats_vs_dogs.h5")

npzfile = numpy.load("cats_vs_dogs_test_images.npz")
x_test = npzfile["arr_0"]

def draw_test(name, pred, input_img):
    BLACK = [0,0,0]
    if pred == "[0]":
        pred = "cat"
    if pred == "[1]":
        pred = "dog"

    expanded_img = cv2.copyMakeBorder(input_img, 0,0,0, inputL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_img, str(pred), (252,70), cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0),2)
    cv2.imshow(name, expanded_img)

for i in range(11):
    rand = numpy.random.randint(0, len(x_test))
    input_img = x_test[rand]

    inputL = cv2.resize(input_img, None, fx =2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("test image", inputL)

    input_img = input_img.reshape(1,150,150,3)

    res = str(classifier.predict_classes(input_img, 1, verbose = 0)[0])

    draw_test("Prediction", res, inputL)
    cv2.waitKey(0)

cv2.destroyAllWindows()

    