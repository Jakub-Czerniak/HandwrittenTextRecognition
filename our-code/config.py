import os
from datetime import datetime

class ModelConfig:
  def __init__(self) -> None:
    self.model_path = os.path.join("models", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
    self.vocabulary = ""
    self.height = 32
    self.width = 128
    self.max_text_length = 0
    self.batch_size = 16
    self.learning_rate = 0.0005
    self.train_epochs = 50