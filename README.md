import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from dataset_single_digit import load_annotations, BatchGenerator, load_validation_data
from dataset_single_digit import convert_to_categorical, create_cls_mapping
import keras.backend as K
from keras.applications import Xception, MobileNet, ResNet50, VGG16
from keras.utils import multi_gpu_model
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam

Running single model:
Download wheights from ::: and extract them into the weights folder

Uses a single resnet50 model for prediciton score should be 0.970.

python3 resnet50_test_single_digit.py

Enesembled model:

Consist of 3 models trained from ResNet50 architecture, and 4 models trained from InceptionResNetV2.

python3 resnet50_test_ensemble_single_digit.py