import sys
from tensorflow import keras
from load_data import load_data
from partition_data import get_siamese_data
from plots.evaluate import *
from utilis import *
import json

train_images, train_labels, test_images, test_labels = load_data(200, 10)
np.save('./data/train_images', train_images)
np.save('./data/train_labels', train_labels)
np.save('./data/test_images', test_images)
np.save('./data/test_labels', test_labels)
print('data loaded')

seen_indexes = np.where(test_labels < 100)[0]
unseen_indexes = np.where(test_labels >= 100)[0]


seen_test_left, seen_test_right, seen_test_ground_truth, seen_test_left_label = get_siamese_data(test_images[seen_indexes], test_labels[seen_indexes], num_classes=100)
unseen_test_left, unseen_test_right, unseen_test_ground_truth, unseen_test_left_label = get_siamese_data(test_images[unseen_indexes], test_labels[unseen_indexes], num_classes=100)

stats = {}

model = keras.models.load_model('saved_models/exp_20_12/' + 'VGG16__model_0_V8', custom_objects={'contrastive_loss': contrastive_loss,
'accuracy1': accuracy1})
model.load_weights('saved_models/exp_20_12/checkpoints/' + 'VGG16__model_0_V8')

file_names = ['VGG16__model_0_V6_SEEN_V2', 'VGG16__model_0_V6_UNSEEN_V2']
for model_name, data in  zip(file_names, [[seen_test_left, seen_test_right, seen_test_ground_truth],[unseen_test_left, unseen_test_right, unseen_test_ground_truth]]):

    test_left = data[0]
    test_right = data[1]
    test_ground_truth = data[2]

    stats[model_name] = {}
    distance, accuracy, precision, recall, false_accept_rate, false_reject_rate = evaluate_test_set(model, (test_left,test_right), test_ground_truth)
    
    print('accuracy')
    print(model_name)
    print(accuracy)
    
    stats[model_name]['labels'] = [float(x) for x in list(test_ground_truth)]
    stats[model_name]['distance'] = [float(x) for x in list(distance)]
    stats[model_name]['accuracy'] = list(accuracy)
    stats[model_name]['precision'] = list(precision)
    stats[model_name]['recall'] = list(recall)
    stats[model_name]['false_accept_rate'] = list(false_accept_rate)
    stats[model_name]['false_reject_rate'] = list(false_reject_rate)

    json.dump(stats, open('./plots/performance/' + model_name, 'w'))

