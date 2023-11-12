import os
import cv2 as cv


def preprocessing_folder(folder_path):
    for image in os.listdir(folder_path):
        preprocessing(image)


def preprocessing(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)


    if not os.path.exists('/PreprocessedImages'):
        os.makedirs('/PreprocessedImages')
    image_name = os.path.basename(os.path.normpath('/folderA/folderB/folderC/folderD/'))
    cv.imwrite('/PreprocessedImages/' + image_name, img)
