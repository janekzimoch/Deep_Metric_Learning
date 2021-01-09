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

    L2_reg, drop_out, LR = 0, 0, 0

    embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5 = 96,2048,2048,32,79,173,422,449

    imgA = Input(shape=(64,64,3))
    imgB = Input(shape=(64,64,3))

    base_model = VGG_16(embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5, L2_reg, drop_out, input_shape=(64,64,3))

    featsA = base_model(imgA)
    featsB = base_model(imgB)

    distance = Lambda(euclidean_distance, name='compute_ED')([featsA, featsB])
    model = Model(inputs=[imgA, imgB], outputs=distance, name='Contrastive')

    model.load_weights('saved_models/exp_20_12/checkpoints/VGG16__model_0_V4')

    # initial_learning_rate = LR
    # lr_schedule = ExponentialDecay(
    # initial_learning_rate,
    # decay_steps=781,
    # decay_rate=0.95,
    # staircase=True)

    model.compile(loss=contrastive_loss, optimizer=keras.optimizers.Adam(0.0005), 
        metrics=[accuracy1])
    
    model.summary()
    return model



def train_siamese(model, model_name, batch_size, epochs, M):

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

    print('Pair mining M: ' + str(M))
    # Generators - random pairing
    # training_generator = DataGenerator_Pairs(partition['train'], labels, **params)
    # validation_generator = DataGenerator_Pairs(partition['validate'], labels, **params)
    # Generators - with mining
    training_generator = DataGenerator_with_mining(partition['train'], labels, train=True, checkpoint_dir=model_name, M=M, **params)
    validation_generator = DataGenerator_with_mining(partition['validate'], labels, train=False, checkpoint_dir=model_name, M=M, **params)


    history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs, 
    verbose=1, callbacks=[model_checkpoint_callback])
    return model, history


def save(model, history, model_name):
    # SAVE FOR LATER USE
    model.save('./saved_models/exp_20_12/{}'.format(model_name))

    hist_csv_file = 'model_history/exp_20_12/{}'.format(model_name)
    json.dump(history.history, open(hist_csv_file, 'w'))



# model_names = ['no_reg'] #, 'L2_only', 'L2_DropOut']
# reg_values = [[0,0,0.0001]] #, [0.0005,0,0.0003], [0.0005,0.2,0.0005]]  # L2, dropout, LR

# for name, reg in zip(model_names, reg_values):
    

for M in [20,40,60,80]:
    model_name = 'VGG16__model_0_V4_test_' + str(M) #_' + name
    exp_values = 0 # reg

    model = siamese_model_VGG(exp_values)
    model, history = train_siamese(model, model_name, batch_size=64, epochs=10, M=M)
    save(model, history, model_name = model_name)
