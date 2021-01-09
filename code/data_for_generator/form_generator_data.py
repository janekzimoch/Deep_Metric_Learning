import numpy as np
import json
from load_data import load_data

train_images, train_labels, test_images, test_labels = load_data(num_classes = 100, num_samples=500)


[np.save('./data_for_generator/c100_s500/{}'.format(id_), image) for image, id_ in zip(train_images, np.arange(len(train_images)))]
[np.save('./data_for_generator/c100_s500/{}'.format(id_), image) for image, id_ in zip(test_images, np.arange(len(train_images),len(train_images)+ len(test_images)))]

labels = np.concatenate([train_labels,test_labels])
np.save('./data_for_generator/c100_s500/labels', labels)