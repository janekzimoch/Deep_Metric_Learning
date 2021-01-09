import numpy as np
import tensorflow.keras as keras
from backbone_architectures import *
from utilis import *

class DataGenerator_with_mining(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(28,28), n_channels=1,
                 n_classes=2, directory='c100_s200', num_classes=100, shuffle=True, train=True, checkpoint_dir='test', M=99):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.directory = directory
        self.shuffle = shuffle
        self.train_size = 100*500
        self.pos_indexes = [np.where(np.array(list(labels.values())) == i)[0] for i in range(num_classes)]
        self.pos_indexes_TRAIN = [np.where(np.array(list(labels.values())[:self.train_size]) == i)[0] for i in range(num_classes)]
        self.neg_indexes = [np.where(np.array(list(labels.values())) != i)[0] for i in range(num_classes)]
        self.train = train
        self.epoch_increment = 5
        self.M = M
        self.five_closest_categories = np.empty((100,self.M))
        self.train_dataset = None
        self.checkpoint_dir = checkpoint_dir # VGG16__18_12__model_2_final_FINE_TUNNING_V2
        self.start_mining = False

        if self.train == True:
            self.train_dataset = np.empty((self.train_size, *self.dim, self.n_channels))
            print('loading data...')
            for ID in range(self.train_size):
                self.train_dataset[ID,:,:,:] = np.load('./data_for_generator/' + self.directory + str(ID) + '.npy')
            print('data loaded, displaying 3 pixels: ' + str(self.train_dataset[0,0,0,:]))

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]  # indexes for mini-batch number: index

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return [X[:,:,:,:,0], X[:,:,:,:,1]], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.epoch_increment += 1

        if self.epoch_increment >= 5 and self.train == True:
            
            print('loading model...')
            # load most recent model 
            model = keras.models.load_model('./saved_models/exp_20_12/VGG16__model_0_V4', custom_objects={'contrastive_loss': contrastive_loss,
            'accuracy1': accuracy1})
            model.load_weights('./saved_models/exp_20_12/checkpoints/VGG16__model_0_V4').expect_partial()

            print('getting embeding vecotrs...')
            # get embeding vectors
            test_model = model.get_layer('LeNet_5')  # i am extracting one CNN pipe of the network
            train_embeddings = test_model.predict(self.train_dataset)

            print('compute embeding average...')
            # compute average embeding for class
            avg_embedding = np.empty((100,96))
            for category in range(100):
                avg_embedding[category,:] = np.mean(train_embeddings[self.pos_indexes_TRAIN[category]], axis=0)

            print('distance matrix...')
            # compute 100x100 category distance matrix
            distance_matrix = np.empty((100,100))
            for category in range(100):
                distance_matrix[category,:] = np.sum(np.square(avg_embedding[category,:] - avg_embedding[:,:]), axis=1).flatten()
                idx = np.argsort(distance_matrix[category,:])[:self.M+1]
                self.five_closest_categories[category,:] = idx[1:]

            print('...finished class ranking')
            self.start_mining = True
            self.epoch_increment = 0

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels, num_inputs)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels, 2))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            left_label = self.labels[ID]
            pos_or_neg = np.random.randint(2, size=1)[0]
            
            if pos_or_neg == 1:  # CHOOSE SIMILAR
                random_ID = ID
                while random_ID == ID:
                    random_ID = np.random.choice(self.pos_indexes[left_label])

            else:  # CHOOSE DIS-SIMILAR
                if self.train == True and self.start_mining == True:
                    dis_similar_label = np.random.choice(self.five_closest_categories[left_label])
                    random_ID = np.random.choice(self.pos_indexes[int(dis_similar_label)])
                else:
                    random_ID = np.random.choice(self.neg_indexes[left_label])


            X[i,:,:,:,0] = np.load('./data_for_generator/' + self.directory + str(ID) + '.npy')  # LEFT image
            X[i,:,:,:,1] = np.load('./data_for_generator/' + self.directory + str(random_ID) + '.npy')  # RIGHT image
            y[i] = pos_or_neg

        return X, y