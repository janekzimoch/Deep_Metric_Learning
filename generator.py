import numpy as np
import tensorflow.keras as keras

class DataGenerator_Pairs(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(28,28), n_channels=1,
                 n_classes=2, directory='c100_s200', num_classes=100, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.directory = directory
        self.shuffle = shuffle
        self.on_epoch_end()
        self.pos_indexes = [np.where(np.array(list(labels.values())) == i)[0] for i in range(num_classes)]
        self.neg_indexes = [np.where(np.array(list(labels.values())) != i)[0] for i in range(num_classes)]

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
                random_ID = np.random.choice(self.neg_indexes[left_label])
            
            X[i,:,:,:,0] = np.load('./data_for_generator/' + self.directory + str(ID) + '.npy')  # LEFT image
            X[i,:,:,:,1] = np.load('./data_for_generator/' + self.directory + str(random_ID) + '.npy')  # RIGHT image
            y[i] = pos_or_neg

        return X, y