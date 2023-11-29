import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.python.keras.layers import Reshape, LSTM

from metrics import CERMetric, WERMetric

from ctcloss import ctc_loss

def model_architecture(input_dim, output_dim, config):
  model = Sequential()

  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_dim, padding="same"))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
  model.add(Dropout(0.2))
  model.add(MaxPooling2D((2, 2)))

  model.add(Reshape((16, 256)))

  model.add(LSTM(256, return_sequences=True))
  model.add(LSTM(128))

  model.add(Dense(output_dim+1, activation='softmax'))

  model.compile(optimizer='adam',
              loss=ctc_loss,
              metrics=[CERMetric(), WERMetric()])
  model.summary()

  return model