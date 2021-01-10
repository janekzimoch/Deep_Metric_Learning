from tensorflow.keras import backend as K
import numpy as np
import json



# FUNCTIONS:
# identity_loss()
# contrastive_loss()
# euclidean_distance()
# triplet_loss()
# triplet_accuracy()
# save()


# We use K. operations rather than numpy, so that we can integrate euclidean distance as a node 
# and Tensorflow can figure out how to do backpropagation

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def contrastive_loss(y, d):
    margin = 1
    return K.mean((y) * K.square(d) + (1-y) * K.square(K.maximum(margin - d, 0)))


def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def triplet_loss(x, alpha = 0.2):
    anchor,positive,negative = x
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss



def precision_m(y_true, y_pred):
  y_pred = K.cast(y_pred < 0.5, y_true.dtype)
  true_positives = (K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = (K.round(K.clip(y_pred, 0, 1)))
  precision = K.mean(K.equal(true_positives, predicted_positives))
  return precision

def f1_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall_m(y_true, y_pred):
  y_pred = K.cast(y_pred < 0.5, y_true.dtype)
  true_positives = (K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = (K.round(K.clip(y_true, 0, 1)))
  recall = K.mean(K.equal(true_positives, possible_positives))
  return recall

def accuracy1(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def accuracy2(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 1.0, y_true.dtype)))

def accuracy3(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 2.0, y_true.dtype)))



def triplet_accuracy(_, y_pred):
    subtraction = K.constant([-1, 1], shape=(2, 1))
    diff =  K.dot(y_pred, subtraction)
    loss = K.maximum(K.sign(diff), K.constant(0))
    return loss


def save(model, history, model_name):
    # SAVE FOR LATER USE
    model.save('./saved_models/16_12/{}'.format(model_name))

    hist_csv_file = 'model_history/16_12/{}'.format(model_name)
    json.dump(history.history, open(hist_csv_file, 'w'))