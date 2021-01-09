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
model = keras.models.load_model('saved_models/exp_20_12/' + 'VGG16__model_0_V8', custom_objects={'contrastive_loss': contrastive_loss,
'accuracy1': accuracy1})
model.load_weights('saved_models/exp_20_12/checkpoints/' + 'VGG16__model_0_V8')

# get embeding network
embedding_model = model.get_layer('LeNet_5')  # i am extracting one CNN pipe of the network


# get THRESHOLD
stats = json.load(open('./plots/performance/VGG16__model_0_V6_SEEN', 'r'))
threshold = ( np.argmax(stats['VGG16__model_0_V6_SEEN']['accuracy']) + 1 ) / 100


# GET LABEL NAMES
label_names = {}
for key in range(200):
    label_names = get_class_to_id_dict()

stats = {}
stats['seen'] = {}
stats['seen']['accuracy'] = {}
stats['seen']['label'] = {}

stats['unseen'] = {}
stats['unseen']['accuracy'] = {}
stats['unseen']['label'] = {}




# accuracy = np.zeros(100)
# false_reject_rate = np.zeros(100)
# false_accept_rate = np.zeros(100)
# label = []
# # SEEN CATEGORIES
# # creates: left, pos_right, neg_right
# for category in range(100):
#     category_indexes = np.where(test_labels == category)[0]

#     pairs = 10
#     size = len(category_indexes)

#     left = np.zeros((pairs*size,64,64,3))
#     pos_right = np.zeros((pairs*size,64,64,3))
#     neg_right = np.zeros((pairs*size,64,64,3))

#     pos_indexes = np.where(test_labels == category)[0]
#     neg_indexes = np.where(test_labels[seen_indexes] != category)[0]

#     ind = 0
#     for index in range(size):
#         for _ in range(pairs):

#             left[ind] = test_images[category_indexes[index],:,:,:]

#             # get POSITIVE PAIRS
#             index_right = category_indexes[index]
#             while index_right == category_indexes[index]:
#                 index_right = np.random.choice(pos_indexes)
#             pos_right[ind] = test_images[index_right,:,:,:]


#             # get NEGATIVE PAIRS
#             index_right = np.random.choice(neg_indexes)
#             neg_right[ind] = test_images[index_right,:,:,:]

#             ind +=1


#     left_embeddings = embedding_model(left)
#     pos_right_embeddings = embedding_model(pos_right)
#     neg_right_embeddings = embedding_model(neg_right)


#     positive_distances = compute_euclidean_distance(left_embeddings, pos_right_embeddings)
#     negative_distances = compute_euclidean_distance(left_embeddings, neg_right_embeddings)
    
#     # plt.hist(positive_distances, bins=30, alpha=0.5, label='similar')
#     # plt.hist(negative_distances, bins=30, alpha=0.5, label='similar')
#     # plt.savefig('./figures/final/trial.png')


#     max_distance = max(max(negative_distances), max(positive_distances))
#     norm_positive_distances = positive_distances / max_distance
#     norm_negative_distances = negative_distances / max_distance


#     TP = np.sum(norm_positive_distances < threshold)
#     FP = np.sum(norm_negative_distances < threshold)
#     TN = np.sum(norm_negative_distances > threshold)
#     FN = np.sum(norm_positive_distances > threshold)
#     false_reject_rate[category] = FN / (TP+FN)
#     false_accept_rate[category] = FP / (TN+FP)
#     accuracy[category] = (TP+TN)/(TP+TN+FP+FN)
#     label += [label_names[category][1].rstrip()]
#     print(str(category) + '-' + label[category] + '   ' + str(accuracy[category]))
#     print('FRR: ' + str(false_reject_rate[category]) + '   ' + 'FAR: ' + str(false_accept_rate[category]))

# label = np.array(label)

# descending_indexes = np.argsort(accuracy)[::-1]
# stats['seen']['accuracy'] = list(accuracy[descending_indexes])
# stats['seen']['false_reject_rate'] = list(false_reject_rate[descending_indexes])
# stats['seen']['false_accept_rate'] = list(false_accept_rate[descending_indexes])
# stats['seen']['label'] = list(label[descending_indexes])

# print('BEST SEEN')
# print(stats['seen']['accuracy'][:5])
# print('FRR' + str(stats['seen']['false_reject_rate'][:5]))
# print('FAR' + str(stats['seen']['false_accept_rate'][:5]))
# print(stats['seen']['label'][:5])
# print('WORST SEEN')
# print(stats['seen']['accuracy'][-5:])
# print('FRR' + str(stats['seen']['false_reject_rate'][-5:]))
# print('FAR' + str(stats['seen']['false_accept_rate'][-5:]))
# print(stats['seen']['label'][-5:])





accuracy = np.zeros(100)
false_reject_rate = np.zeros(100)
false_accept_rate = np.zeros(100)
label = []
# SEEN CATEGORIES
# creates: left, pos_right, neg_right
for category in range(100,200):
    category_indexes = np.where(test_labels == category)[0]

    pairs = 10
    size = len(category_indexes)

    left = np.zeros((pairs*size,64,64,3))
    pos_right = np.zeros((pairs*size,64,64,3))
    neg_right = np.zeros((pairs*size,64,64,3))

    pos_indexes = np.where(test_labels == category)[0]
    neg_indexes = np.where(test_labels[unseen_indexes] != category)[0]

    ind = 0
    for index in range(size):
        for _ in range(pairs):

            left[ind] = test_images[category_indexes[index],:,:,:]

            # get POSITIVE PAIRS
            index_right = category_indexes[index]
            while index_right == category_indexes[index]:
                index_right = np.random.choice(pos_indexes)
            pos_right[ind] = test_images[index_right,:,:,:]


            # get NEGATIVE PAIRS
            index_right = np.random.choice(neg_indexes)
            neg_right[ind] = test_images[index_right,:,:,:]

            ind +=1


    left_embeddings = embedding_model(left)
    pos_right_embeddings = embedding_model(pos_right)
    neg_right_embeddings = embedding_model(neg_right)

    positive_distances = compute_euclidean_distance(left_embeddings, pos_right_embeddings)
    negative_distances = compute_euclidean_distance(left_embeddings, neg_right_embeddings)
    
    # plt.hist(positive_distances, bins=30, alpha=0.5, label='similar')
    # plt.hist(negative_distances, bins=30, alpha=0.5, label='similar')
    # plt.savefig('./figures/final/trial.png')


    max_distance = max(max(negative_distances), max(positive_distances))
    norm_positive_distances = positive_distances / max_distance
    norm_negative_distances = negative_distances / max_distance


    TP = np.sum(norm_positive_distances < threshold)
    FP = np.sum(norm_negative_distances < threshold)
    TN = np.sum(norm_negative_distances > threshold)
    FN = np.sum(norm_positive_distances > threshold)
    false_reject_rate[category-100] = FN / (TP+FN)
    false_accept_rate[category-100] = FP / (TN+FP)
    accuracy[category-100] = (TP+TN)/(TP+TN+FP+FN)
    label += [label_names[category][1].rstrip()]
    print(str(category) + '-' + label[category-100] + '   ' + str(accuracy[category-100]))
    print('FRR: ' + str(false_reject_rate[category-100]) + '   ' + 'FAR: ' + str(false_accept_rate[category-100]))


label = np.array(label)

descending_indexes = np.argsort(accuracy)[::-1]
stats['unseen']['accuracy'] = list(accuracy[descending_indexes])
stats['unseen']['false_reject_rate'] = list(false_reject_rate[descending_indexes])
stats['unseen']['false_accept_rate'] = list(false_accept_rate[descending_indexes])
stats['unseen']['label'] = list(label[descending_indexes])
print('BEST UNSEEN')
print(stats['unseen']['accuracy'][:5])
print('FRR' + str(stats['unseen']['false_reject_rate'][:5]))
print('FAR' + str(stats['unseen']['false_accept_rate'][:5]))
print(stats['unseen']['label'][:5])
print('WORST UNSEEN')
print(stats['unseen']['accuracy'][-5:])
print('FRR' + str(stats['unseen']['false_reject_rate'][-5:]))
print('FAR' + str(stats['unseen']['false_accept_rate'][-5:]))
print(stats['unseen']['label'][-5:])


json.dump(stats, open('./plots/performance/stats', 'w'))
