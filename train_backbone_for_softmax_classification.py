import numpy as np
import json
import random
import matplotlib.pyplot as plt

from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import AveragePooling2D,Input,Layer,Dense, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, Convolution2D, MaxPooling2D, concatenate, Lambda, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import initializers

from load_data import load_data
from partition_data import get_siamese_data, get_triplet_data
from backbone_architectures import *
from utilis import *

np.random.seed(123)  # for reproducibility

# LOAD SUBSET OF DATA
train_images = np.load('./data/train_images.npy')
train_labels = np.load('./data/train_labels.npy')
test_images = np.load('./data/test_images.npy')
test_labels = np.load('./data/test_labels.npy')
print('data loaded')

# LOAD DATA
# train_left, train_right, train_ground_truth, train_left_label = get_siamese_data(train_images, train_labels, num_classes=20)
# test_left, test_right, test_ground_truth, test_left_label = get_siamese_data(test_images, test_labels, num_classes=20)

# np.save('./data/backbone/train_left', train_left)
# np.save('./data/backbone/train_right', train_right)
# np.save('./data/backbone/train_ground_truth', train_ground_truth)
# np.save('./data/backbone/train_left_label', train_left_label)

# np.save('./data/backbone/test_left', test_left)
# np.save('./data/backbone/test_right', test_right)
# np.save('./data/backbone/test_ground_truth', test_ground_truth)
# np.save('./data/backbone/test_left_label', test_left_label)
# print('partitioned data saved')

# train_left = np.load('./data/backbone/train_left.npy')
# train_right = np.load('./data/backbone/train_right.npy')
# train_ground_truth = np.load('./data/backbone/train_ground_truth.npy')
# train_left_label = np.load('./data/backbone/train_left_label.npy')

# test_left = np.load('./data/backbone/test_left.npy')
# test_right = np.load('./data/backbone/test_right.npy')
# test_ground_truth = np.load('./data/backbone/test_ground_truth.npy')
# test_left_label = np.load('./data/backbone/test_left_label.npy')

# print('partitioned data loaded')

# train backbone architecture
model_backbone = VGG_16()

image = Input(shape=(64,64,3))
image_embeding = model_backbone(image)
classification = Dense(20, activation="softmax", name="20-classifier")(image_embeding)

model_pretrained = Model(inputs=image, outputs=classification, name='pretrained')

boundaries = [1000, 3000]
values = [5e-3, 1e-3, 5e-4]
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

model_pretrained.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr_schedule), metrics=["accuracy"])

model_pretrained.fit(train_images[:3200], train_labels[:3200],
        validation_data=(test_images[:32], test_labels[:32]), batch_size=128, epochs=1000, verbose=1)

model_name = 'VGG_16'
model_pretrained.save('./saved_models/backbone/{}.h5'.format(model_name))
