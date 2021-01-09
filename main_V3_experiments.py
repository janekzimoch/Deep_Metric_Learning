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
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from load_data import load_data
from partition_data import get_siamese_data, get_triplet_data
from backbone_architectures import *
from utilis import *
from generator import DataGenerator_Pairs
from generator_with_mining import DataGenerator_with_mining

np.random.seed(123)  # for reproducibility





num_samples = 500
num_classes = 100
num_train = num_samples*num_classes
num_val = 5000
data_directory = 'c100_s500/'

# DATA ID
partition = {}
labels = {}
partition['train'] = np.arange(num_train)  # list of IDs of trainign data
partition['validate'] = np.arange(num_train, num_train + num_val)  # list of IDs of validation data
label_data = np.load('./data_for_generator/{}labels.npy'.format(data_directory))
for ID, label in zip(range(num_train + num_val), label_data):
    labels[ID] = label



# BUILD SIAMESE NETWORK
def siamese_model_VGG(exp_values):

    embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5 = exp_values

    imgA = Input(shape=(64,64,3))
    imgB = Input(shape=(64,64,3))

    base_model = VGG_16(embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5, input_shape=(64,64,3))

    featsA = base_model(imgA)
    featsB = base_model(imgB)

    distance = Lambda(euclidean_distance, name='compute_ED')([featsA, featsB])
    model = Model(inputs=[imgA, imgB], outputs=distance, name='Contrastive')

    # model.load_weights('saved_models/checkpoints/' + model_name)

    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=781,
    decay_rate=0.90,
    staircase=True)

    model.compile(loss=contrastive_loss, optimizer=keras.optimizers.Adam(lr_schedule), 
        metrics=[accuracy1])
    
    model.summary()
    return model



# BUILD SIAMESE NETWORK
def siamese_model_ResNet(model_type):

    if model_type == 'ResNet_50':
        LR = 0.001
        base_model = ResNet_50()
    if model_type == 'ResNet_18':
        LR = 0.0001
        base_model = ResNet_18()

    imgA = Input(shape=(64,64,3))
    imgB = Input(shape=(64,64,3))

    featsA = base_model(imgA)
    featsB = base_model(imgB)

    distance = Lambda(euclidean_distance, name='compute_ED')([featsA, featsB])
    model = Model(inputs=[imgA, imgB], outputs=distance, name='Contrastive')

    # model.load_weights('saved_models/checkpoints/' + model_name)

    initial_learning_rate = LR
    lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=1562,
    decay_rate=0.90,
    staircase=True)

    model.compile(loss=contrastive_loss, optimizer=keras.optimizers.Adam(lr_schedule), 
        metrics=[accuracy1]) #, recall_m, precision_m])
    
    model.summary()
    return model



def train_siamese(model, model_name, batch_size, epochs):

    params = {'dim': (64,64),
        'batch_size': batch_size,
        'n_classes': 2,
        'n_channels': 3,
        'num_classes': 100,
        'directory': data_directory,
        'shuffle': True}

    checkpoint_filepath = './saved_models/exp_20_12/checkpoints/' + model_name 
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Generators - random pairing
    training_generator = DataGenerator_Pairs(partition['train'], labels, **params)
    validation_generator = DataGenerator_Pairs(partition['validate'], labels, **params)
    # Generators - with mining
    # training_generator = DataGenerator_with_mining(partition['train'], labels, train=True, checkpoint_dir=model_name, **params)
    # validation_generator = DataGenerator_with_mining(partition['validate'], labels, train=False, checkpoint_dir=model_name, **params)


    history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, 
    verbose=1, callbacks=[model_checkpoint_callback])
    return model, history


def save(model, history, model_name):
    # SAVE FOR LATER USE
    model.save('./saved_models/exp_20_12/{}'.format(model_name))

    hist_csv_file = 'model_history/exp_20_12/{}'.format(model_name)
    json.dump(history.history, open(hist_csv_file, 'w'))



# ResNet experiments
for model_type in ['ResNet_18', 'ResNet_50']:

    model_name = '{}__model_'.format(model_type)

    model = siamese_model_ResNet(model_type)
    
    if model_type == 'ResNet_50':
        model, history = train_siamese(model, model_name, batch_size=32, epochs=8)
        save(model, history, model_name = model_name)

    if model_type == 'ResNet_18':
        model, history = train_siamese(model, model_name, batch_size=32, epochs=20)
        save(model, history, model_name = model_name)






# VGG experiments
Random_search = {}
Random_search['fc1'] = [1024, 2048]
Random_search['fc2'] = [1024, 2048]
Random_search['embedding_layer'] = [96,128,160]
Random_search['feat1'] = [32,64]
Random_search['feat2'] = [128]
Random_search['feat3'] = [256]
Random_search['feat4'] = [512]
Random_search['feat5'] = [512]

# for model_num in range(6):

#     embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5 = 0,0,0,0,0,0,0,0

#     for key, parameters in Random_search.items():
#         if key == 'embedding_layer':
#             embedding_length = np.random.choice(parameters)

#         if key == 'fc1':
#             fc1 = np.random.choice(parameters)
#         if key == 'fc2':
#             fc2 = np.random.choice(parameters)

#         if key == 'feat1':
#             feat1 = np.random.choice(parameters)

#         if key == 'feat2':
#             feat2 = int(np.random.uniform(low=feat1, high=parameters))
#         if key == 'feat3':
#             feat3 = int(np.random.uniform(low=feat2, high=parameters))
#         if key == 'feat4':
#             feat4 = int(np.random.uniform(low=feat3, high=parameters))
#         if key == 'feat5':
#             feat5 = int(np.random.uniform(low=feat4, high=parameters))

# model_num = 6
# embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5 = 96,2048,2048,32,79,173,422,449
# exp_values = [embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5]
# exp_meaning = ['embedding_length', 'fc1', 'fc2', 'feat1','feat2','feat3','feat4','feat5']
# print('PARAMS:')
# print(exp_values)

# with open('./model_history/exp_20_12/model_list.txt', 'a') as f:
#     f.write("Model {}: ".format(model_num))
#     for meaning, item in zip(exp_meaning, exp_values):
#         f.write(meaning + ': ' + str(item) + ',  ')
#     f.write("\n")


# exp_name = 'model_{}'.format(model_num)
# exp_values = [embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5]
# model_name = 'VGG16__model_' + str(model_num)

# model = siamese_model_VGG(exp_values)
# model, history = train_siamese(model, model_name, batch_size=64, epochs=20)
# save(model, history, model_name = model_name)
