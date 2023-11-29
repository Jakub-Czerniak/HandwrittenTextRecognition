import numpy as np
import cv2
import copy

class DataProvider():
    def __init__(self, x, y, img_width=200, img_height=200):
        self.x = x
        self.y = y
        self.width = img_width
        self.height = img_height

    def prepare_data(self):
        images = []
        for file_name in self.x:
            img = cv2.resize(cv2.imread(file_name), (self.width, self.height))
            img = np.array(img)
            images.append(img)
        labels = []
        for y in self.y:
            labels.append(np.array(y))
        
        labels = np.array(labels)

        return np.array(images), labels
    
    def split(self, split: float):
        train_data_provider, val_data_provider = copy.deepcopy(self), copy.deepcopy(self)

        train_data_provider.x = self.x[:int(len(self.x) * split)]
        train_data_provider.y = self.y[:int(len(self.y) * split)]
        val_data_provider.x = self.x[int(len(self.x) * split):]
        val_data_provider.y = self.y[int(len(self.y) * split):]

        return train_data_provider, val_data_provider