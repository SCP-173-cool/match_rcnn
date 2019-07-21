# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:15:50 2019

@author: loktarxiao
"""

from lib.config import Config
from lib import model as modellib

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


config = BalloonConfig()

model = modellib.MaskRCNN(mode="training", config=config, model_dir="./logs/")