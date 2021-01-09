import sys
from tensorflow import keras
from plots.evaluate import *
from utilis import *
import json

# exp_name = 'embedding_layer_V2'
# file_names = ['4096-4096', '2048-4096', '2048-2048', '1024-2048', '1024-1024']
# file_names = ['16_V2', '32_V2', '64_V2', '128_V2', '256_V2', '512_V2']
# file_names = ['64-128-256-512-512', '32-64-128-256-512', '64-64-128-128-256']
# file_names = ['VGG_16_final_architecture']
# file_names = ['model_0_final', 'model_1_final', 'model_2_final', 'model_3_final', 'model_4_final']
# file_names = ['model_2_final_FINE_TUNNING']
# file_names = ['ResNet_50__model_', 'ResNet_18__model_', 'VGG16__model_0']
file_names = ['VGG16__model_0_V5']
stats = {}

test_left = np.load('./data/partitioned/test_left.npy')
test_right = np.load('./data/partitioned/test_right.npy')
test_ground_truth = np.load('./data/partitioned/test_ground_truth.npy')

for model_name in  file_names:

    model = keras.models.load_model('saved_models/exp_20_12/' + model_name, custom_objects={'contrastive_loss': contrastive_loss,
    'accuracy1': accuracy1})
    
    model.load_weights('saved_models/exp_20_12/checkpoints/' + model_name)

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

json.dump(stats, open('./plots/performance/VGG16__20_12__final_V5', 'w'))

