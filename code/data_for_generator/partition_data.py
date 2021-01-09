import numpy as np
np.random.seed(123)  # for reproducibility


def form_data_pairs(images, labels, pairs=4, num_classes=20):
    
    size = len(labels)
    lowest_label = min(labels)
    highest_label = max(labels)

    left_input = np.zeros((2*pairs*size,64,64,3))
    right_input = np.zeros((2*pairs*size,64,64,3))
    ground_truth = np.zeros((2*pairs*size))
    left_labels = np.zeros((2*pairs*size))
    
    indexes = [np.where(labels == i)[0] for i in range(highest_label+1)]
    neg_indexes = [np.where(labels != i)[0] for i in range(highest_label+1)]
    
    #Let's create the new dataset to train on
    ind = 0
    for left_index in range(size):
        
        # SIMILAR pairs
        for _ in range(pairs):
            label = labels[left_index]
            right_index = np.random.choice(indexes[label])
            
            left_input[ind] = images[left_index,:,:,:]
            right_input[ind] = images[right_index,:,:,:]
            ground_truth[ind] = 1.
            left_labels[ind] = label


            ind += 1  # first training pair collected
            
        # DISSIMILAR pairs
        for _ in range(pairs):
            label = labels[left_index]
            right_index = np.random.choice(neg_indexes[label])
            
            left_input[ind] = images[left_index,:,:,:]
            right_input[ind] = images[right_index,:,:,:]
            ground_truth[ind] = 0.
            left_labels[ind] = label


            ind += 1  # first training pair collected
        
    # shuffle data 
    new_ordering = np.arange(size*2*pairs)
    np.random.shuffle(new_ordering)

    left_input = left_input[new_ordering,:,:,:]
    right_input = right_input[new_ordering,:,:,:]
    ground_truth = ground_truth[new_ordering]
    left_labels = left_labels[new_ordering]
    
    return left_input, right_input, ground_truth, left_labels



def form_data_triplets(images, labels, pairs=5, num_classes=20):
    
    size = len(labels)

    list_a = np.zeros((2*pairs*size,64,64,3))
    list_n = np.zeros((2*pairs*size,64,64,3))
    list_p = np.zeros((2*pairs*size,64,64,3))
    
    pos_indexes = [np.where(labels == i)[0] for i in range(num_classes)]
    neg_indexes = [np.where(labels != i)[0] for i in range(num_classes)]
    
    ind = 0
    for index_a in range(size):
        
        for _ in range(2*pairs):
            # ANCHOR
            label = labels[index_a]
            list_a[ind] = images[index_a,:,:,:] 

            # NEGATIVE
            index_n = np.random.choice(neg_indexes[label])
            list_n[ind] = images[index_n,:,:,:]

            # POSITIVE
            index_p = np.random.choice(pos_indexes[label])
            list_p[ind] = images[index_p,:,:,:]

            ind += 1

    # shuffle data 
    new_ordering = np.arange(2*size*pairs)
    np.random.shuffle(new_ordering)

    list_a = list_a[new_ordering,:,:,:]
    list_n = list_n[new_ordering,:,:,:]
    list_p = list_p[new_ordering]
    
    return list_a, list_n, list_p




def get_siamese_data(images, labels, pairs=4, num_classes=20):

    left, right, ground_truth, left_label = form_data_pairs(images, labels, pairs=pairs, num_classes=num_classes)
    print('Fraction true: ' + str(ground_truth.sum()/len(ground_truth)) + '  total num: ' + str(len(ground_truth)))
    return left, right, ground_truth, left_label


def get_triplet_data(images, labels, num_classes=20):

    a, n, p = form_data_triplets(images, labels, num_classes=num_classes)
    return a, n, p