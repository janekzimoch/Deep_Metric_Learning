from tensorflow import keras
from load_data import load_data
from partition_data import get_siamese_data
from plots.evaluate import *
from utilis import *
from load_data import get_class_to_id_dict
import matplotlib.pyplot as plt
import json


test_images = np.load('./data/test_images.npy')
test_labels = np.load('./data/test_labels.npy')
print(test_labels)
print('data loaded')

seen_indexes = np.where(test_labels < 100)[0]
unseen_indexes = np.where(test_labels >= 100)[0]


# Load model
model = keras.models.load_model('saved_models/exp_20_12/' + 'VGG16__model_0_V6', custom_objects={'contrastive_loss': contrastive_loss,
'accuracy1': accuracy1})
model.load_weights('saved_models/exp_20_12/checkpoints/' + 'VGG16__model_0_V6')

# get embeding network
embedding_model = model.get_layer('LeNet_5')  # i am extracting one CNN pipe of the network


seen_embeddings = embedding_model.predict(test_images[seen_indexes[:]])
unseen_embeddings = embedding_model.predict(test_images[seen_indexes[:]])


np.save('./data/seen_embeddings', seen_embeddings)
np.save('./data/unseen_embeddings', unseen_embeddings)
np.save('./data/seen_labels', test_labels[seen_indexes])
np.save('./data/unseen_labels', test_labels[unseen_indexes])

# json.dump(embeddings, open('./plots/performance/embeddings', 'w'))
