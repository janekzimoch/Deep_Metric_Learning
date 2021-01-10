import numpy as np
from utilis import *


np.random.seed(123)  # for reproducibility




def compute_cosine_distance(left_target_vector, right_target_vector):
    cosine_distance = np.sum(np.multiply(left_target_vector,right_target_vector), axis=1)/(np.linalg.norm(left_target_vector, axis=1) * np.linalg.norm(right_target_vector, axis=1))
    return cosine_distance

def compute_euclidean_distance(left_target_vector, right_target_vector):
    euclidean_distance = np.sqrt(np.sum(np.power(left_target_vector - right_target_vector,2), axis=1))
    return euclidean_distance

def compute_squared_euclidean_distance(left_target_vector, right_target_vector):
    euclidean_distance = np.sum(np.power(left_target_vector - right_target_vector,2), axis=1)
    return euclidean_distance


# plots distribution of EUCLIDEAN DISTANCES for similar and dis-similar classes
def get_embeding_vectors(dataset, model, layer='LeNet_5'):
    test_model = model.get_layer('LeNet_5')  # i am extracting one CNN pipe of the network
    vectors = test_model.predict(dataset)
    return vectors


def get_distance(model, images, distance='euclidean'):
    
    img_L, img_R = images

    testData_embeding_Left = get_embeding_vectors(img_L, model)
    testData_embeding_Right = get_embeding_vectors(img_R, model)
    
    if distance == 'euclidean':
        testData_distance = compute_euclidean_distance(testData_embeding_Left, testData_embeding_Right)
    if distance == 'cosine':
        testData_distance = compute_cosine_distance(testData_embeding_Left, testData_embeding_Right)
    if distance == 'sq_euclidean':
        testData_distance = compute_squared_euclidean_distance(testData_embeding_Left, testData_embeding_Right)
        
    return testData_distance



def evaluate_test_set(model, images, labels):
    
    distance = get_distance(model, images, distance='euclidean')
    distance_normalised = distance / np.max(distance)
    
    list_thr = np.arange(0,1,0.01)
    accuracy = np.zeros(len(list_thr))
    precision = np.zeros(len(list_thr)) 
    recall = np.zeros(len(list_thr))
    false_accept_rate = np.zeros(len(list_thr)) 
    false_reject_rate = np.zeros(len(list_thr)) 
    
    for i, threshold in enumerate(list_thr):
        FP, FN, TP, TN = get_confusion_matrix(distance_normalised, labels, threshold)
        
        accuracy[i] = (TP+TN) / (TP+TN+FP+FN)
        precision[i] = TP / (TP+FP+0.00001)
        recall[i] = TP / (TP+FN+0.00001)  # same as VALIDATION RATE
        false_accept_rate[i] = FP / (TN+FP+0.00001)
        false_reject_rate[i] = FN / (TP+FN+0.00001)
        

    return distance, accuracy, precision, recall, false_accept_rate, false_reject_rate


def get_confusion_matrix(distance, labels, threshold):
    
    num_pairs = len(labels)
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    
    y_pred = [1. if i<threshold else 0. for i in distance]
    
    for i in range(len(labels)):
        if labels[i] == 1: # IMAGES SIMILAR
            if y_pred[i] == 1:
                TP +=1
            else:
                FN +=1
        else:  # IMAGES DISSIMILAR
            if y_pred[i] == 0:
                TN +=1
            else:
                FP +=1
    
    return FP, FN, TP, TN