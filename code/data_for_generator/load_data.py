import time
# import matplotlib.pyplot as plt
import cv2
import numpy as np
np.random.seed(123)  # for reproducibility

path = 'tiny-imagenet-200/'

def get_id_dictionary():
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict():
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

def get_data(id_dict, num_samples):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [cv2.cvtColor(cv2.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i))), cv2.COLOR_BGR2RGB) for i in range(num_samples)]
        train_labels_ = np.array([[0]*200]*num_samples)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(cv2.cvtColor(cv2.imread( path + 'val/images/{}'.format(img_name)), cv2.COLOR_BGR2RGB))
        test_labels_ = np.array([[0]*200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def shuffle_data(images, labels):
    new_ordering = np.arange(len(labels))
    np.random.shuffle(new_ordering)
    
    images = images[new_ordering,:,:,:]
    labels = labels[new_ordering]
    return images, labels


def load_data(num_classes, num_samples):

    train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary(), num_samples)

    print("ALL DATA")
    print( "train data shape: ",  train_data.shape )
    print( "train label shape: ", train_labels.shape )
    print( "test data shape: ",   test_data.shape )
    print( "test_labels.shape: ", test_labels.shape )

    train_labels = np.argmax(train_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    train_indexes = [i for i, val in enumerate(train_labels) if val in np.arange(num_classes)] 
    test_indexes = [i for i, val in enumerate(test_labels) if val in np.arange(num_classes)] 

    train_images, train_labels, test_images, test_labels = train_data[train_indexes], train_labels[train_indexes], test_data[test_indexes], test_labels[test_indexes]


    # train_images, train_labels = shuffle_data(train_images, train_labels)
    # test_images, test_labels = shuffle_data(test_images, test_labels)

    train_images = train_images.astype('float32')/255.0
    test_images = test_images.astype('float32')/255.0

    # get mean pixel values
    mean = np.mean(train_images, axis=0)
    train_images = train_images - mean
    test_images = test_images - mean


    print("50-CLASS DATA ONLY")
    print( "train data shape: ",  train_images.shape )
    print( "train label shape: ", train_labels.shape )
    print( "test data shape: ",   test_images.shape )
    print( "test_labels.shape: ", test_labels.shape )

    return train_images, train_labels, test_images, test_labels