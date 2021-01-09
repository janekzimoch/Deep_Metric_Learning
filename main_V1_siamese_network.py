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


# LOAD DATA - takes 3-5 minutes
# train_images, train_labels, test_images, test_labels = load_data()
# np.save('./data/train_images', train_images)
# np.save('./data/train_labels', train_labels)
# np.save('./data/test_images', test_images)
# np.save('./data/test_labels', test_labels)



# LOAD SUBSET OF DATA
train_images = np.load('./data/train_images.npy')
train_labels = np.load('./data/train_labels.npy')
test_images = np.load('./data/test_images.npy')
test_labels = np.load('./data/test_labels.npy')
print('data loaded')




# PARTITION DATA
# pairs   
# train_left, train_right, train_ground_truth, train_left_label = get_siamese_data(train_images, train_labels, num_classes=20)
# test_left, test_right, test_ground_truth, test_left_label = get_siamese_data(test_images, test_labels, num_classes=20)

# np.save('./data/partitioned_small/train_left', train_left)
# np.save('./data/partitioned_small/train_right', train_right)
# np.save('./data/partitioned_small/train_ground_truth', train_ground_truth)
# np.save('./data/partitioned_small/test_left', test_left)
# np.save('./data/partitioned_small/test_right', test_right)
# np.save('./data/partitioned_small/test_ground_truth', test_ground_truth)
# print('partitioned data saved')

# train_left = np.load('./data/partitioned_small/train_left.npy')
# train_right = np.load('./data/partitioned_small/train_right.npy')
# train_ground_truth = np.load('./data/partitioned_small/train_ground_truth.npy')
# test_left = np.load('./data/partitioned_small/test_left.npy')
# test_right = np.load('./data/partitioned_small/test_right.npy')
# test_ground_truth = np.load('./data/partitioned_small/test_ground_truth.npy')
# print('partitioned data loaded')

# print(train_ground_truth[:32])

# plt.imshow(train_left[0,:,:,:])
# plt.savefig('./figures/data_check.png')
# print('figure saved')

# triplets
train_a, train_n, train_p = get_triplet_data(train_images[:], train_labels[:], num_classes=20)
test_a, test_n, test_p = get_triplet_data(test_images, test_labels, num_classes=20)



# BUILD SIAMESE NETWORK
def siamese_model(loss=contrastive_loss, model_name='contrastive_loss', fine_tune=True):
    imgA = Input(shape=(64,64,3))
    imgB = Input(shape=(64,64,3))
    # base_model = LeNet_5((64,64,3))
    # base_model = ResNet_50()
    base_model = VGG_16()
    # base_model = VGG_16_mini()

    featsA = base_model(imgA)
    featsB = base_model(imgB)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='compute_ED')([featsA, featsB])
    normalized_distance = BatchNormalization()(distance)
    outputs = Dense(1, activation="sigmoid", name='Sigmoid_classification')(normalized_distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs, name=model_name)

        # load weights
    # model_file = 'VGG_16_zero_centered_data_CONTRASTIVE_15_12_20__V4'
    # loaded_model = keras.models.load_model('saved_models/16_12/' +model_file, custom_objects={'contrastive_loss': contrastive_loss,
    # 'accuracy1': accuracy1})
    # model.set_weights(loaded_model.get_weights())
    

    # boundaries = [3000, 6000]
    # values = [1e-2, 1e-3, 5e-4]
    # lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(0.005),
        metrics=["accuracy"])
    
    model.summary()
    return model



# BUILD SIAMESE NETWORK
def siamese_model_CONTR(loss=contrastive_loss, model_name='contrastive_loss', fine_tune=True):
    imgA = Input(shape=(64,64,3))
    imgB = Input(shape=(64,64,3))
    # base_model = LeNet_5((64,64,3))
    # base_model = ResNet_50()
    base_model = VGG_16()

    featsA = base_model(imgA)
    featsB = base_model(imgB)

    distance = Lambda(euclidean_distance, name='compute_ED')([featsA, featsB])
    model = Model(inputs=[imgA, imgB], outputs=distance, name=model_name)
    
    # load weights
    model_file = 'VGG_16_zero_centered_data_CONTRASTIVE_15_12_20__V4'
    loaded_model = keras.models.load_model('saved_models/16_12/' +model_file, custom_objects={'contrastive_loss': contrastive_loss,
    'accuracy1': accuracy1})
    model.set_weights(loaded_model.get_weights())

    # boundaries = [2500]
    # values = [1e-3, 5e-4]
    # lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(0.0005), 
        metrics=[accuracy1]) #, recall_m, precision_m])
    
    model.summary()
    return model


def train_siamese(model, model_name='test'):
    history = model.fit(
        [train_left[:],train_right[:]], train_ground_truth[:],
        validation_data=([test_left[:],test_right[:]],test_ground_truth[:]), batch_size=254, epochs=3, verbose=1)
    
    return model, history



# BUILD TRIPLET NETWORK
def triplet_model(fine_tune=True):
    input_1 = Input(shape=(64,64,3))
    input_2 = Input(shape=(64,64,3))
    input_3 = Input(shape=(64,64,3))
    
    # base_model = LeNet_5((64,64,3))
    base_model = VGG_16()

    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)
   
    loss = Lambda(triplet_loss, name='compute_ED')([A, P, N]) 
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss, name='triplet_loss')
    
    # if fine_tune:
    #     model.set_weights(models_storage['LeNet_5_triplet_v1'].get_weights())
    
    model.compile(loss=identity_loss, optimizer=keras.optimizers.Adam(0.0005))

    model.summary()
    return model

def train_triplet(model, model_name='test'):
    history = model.fit(
        [train_a[:], train_p[:], train_n[:]], np.zeros(len(train_n[:])),
        validation_data=([test_a[:], test_p[:], test_n[:]], np.zeros(len(train_n[:]))), batch_size=64, epochs=10, verbose=1)
    
    return model, history




# RUN
# CROSS-ENTROPY
# model = siamese_model(loss="binary_crossentropy", model_name='cross-entropy_loss', fine_tune=False)
# model, history = train_siamese(model, model_name='VGG_16_crossentropy_v1')
# save(model, history, model_name='VGG_16_Xentropy_16_12_20__V1')

# # CONTRASTIVE
# model = siamese_model_CONTR(loss=contrastive_loss, model_name='contrastive_loss', fine_tune=False)
# model, history = train_siamese(model, model_name='VGG_16_contrastive_v1')
# save(model, history, model_name='VGG_16_500k_regularised_CONTRASTIVE_15_12_20__V5')

# TRIPLET
model = triplet_model(fine_tune=False)
model, history = train_triplet(model, model_name='LeNet_5_triplet_v1')
# save(model, history, model_name='LeNet_5_TRIPLET_14_12_20_V2')