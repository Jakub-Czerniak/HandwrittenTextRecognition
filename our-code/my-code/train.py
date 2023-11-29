#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm
from config import ModelConfig

from provider import DataProvider

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
import tensorflow as tf

# for unknown reason it's impossible to import CVImage from mltu
# That is why its source code was copied to this project

from mltu_image import CVImage
from model import model_architecture
from metrics import CERMetric, WERMetric
from ctcloss import ctc_loss


project_dir = os.path.abspath("../")
dataset_path = os.path.join(project_dir, "data")

images_names, labels, vocab, max_len = [], [], set(), 0

words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[1] == "err":
        continue

    folder1 = line_split[0][:3]
    folder2 = "-".join(line_split[0].split("-")[:2])
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip("\n")

    rel_path = os.path.join(dataset_path, "words", folder1, folder2, file_name)
    if not os.path.exists(rel_path):
        print(f"File not found: {rel_path}")
        continue

    images_names.append(rel_path)
    labels.append(label)
    vocab.update(list(label))
    max_len = max(max_len, len(label))

config = ModelConfig()
config.vocabulary = "".join(vocab)
config.max_text_length = max_len

data_provider = DataProvider(x=images_names, 
                             y=labels,
                             img_width=config.width,
                             img_height=config.height)

train_data_provider, val_data_provider = data_provider.split(split = 0.9)

model = model_architecture( input_dim = (config.height, config.width, 3),
                            output_dim = len(config.vocabulary), config=config)

checkpoint = ModelCheckpoint(f"{config.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")

x_train, y_train = train_data_provider.prepare_data()

print("INFO: All training images prepared")

model.fit(
    x = x_train,
    y = y_train,
    validation_data=val_data_provider.prepare_data(),
    epochs=config.train_epochs,
    batch_size=config.batch_size,
    callbacks=[checkpoint],
)

train_data_provider.to_csv(os.path.join(config.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(config.model_path, "val.csv"))

# TEST

trained_model = load_model(os.path.join(config.model_path, "model.h5"), 
                           custom_objects= {"ctc_loss" : ctc_loss})