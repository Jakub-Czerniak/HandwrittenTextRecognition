import os
import cv2 as cv
import numpy as np


def preprocessing_folder(folder_path):
    for image in os.listdir(folder_path):
        preprocessing(folder_path + image)


def show_image(img):
    cv.imshow('img', img)
    cv.waitKey()


def preprocessing(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 251, 25)
    # _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # img = cv.dilate(img, np.ones((3, 3)), iterations=1)
    img = cv.fastNlMeansDenoising(src=img, h=10, templateWindowSize=7, searchWindowSize=15)
    if not os.path.exists('PreprocessedImages/'):
        os.makedirs('PreprocessedImages/')
    image_name = os.path.basename(os.path.normpath(image_path))
    cv.imwrite('PreprocessedImages/' + image_name, img)


preprocessing_folder('Images/')
