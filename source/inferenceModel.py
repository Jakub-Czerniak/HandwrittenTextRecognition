#!/usr/bin/env python3

import cv2
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs
from argparse import ArgumentParser

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


parser = ArgumentParser()
parser.add_argument("--model", help="model config file")
parser.add_argument("--data", help="validation data CSV")
args = parser.parse_args()

model = args.model
data = args.data

configs = BaseModelConfigs.load(model)

model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

df = pd.read_csv(data).values.tolist()

accum_cer = []
for image_path, label in tqdm(df):
    image = cv2.imread(image_path)

    prediction_text = model.predict(image)

    cer = get_cer(prediction_text, label)
    print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

    accum_cer.append(cer)

    # resize by 4x
    image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(f"Average CER: {np.average(accum_cer)}")