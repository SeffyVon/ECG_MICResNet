import numpy as np
import matplotlib.pyplot as plt 

def cal_multilabel_weights(Data_labels, threshold_percentile=0):
    """
    Encode old label (multi-hot encoding) to new label (number) with a threshold (>=80-percentile length)
    
    input: Code_label_train (multi-hot encoding)
    output: sample_weights
    """
    # key: multi-hot encoding label in str
    # val: idx

    labels_dict = {}
    for idx, data_label in enumerate(Data_labels):
        key = ''.join([str(l) for l in data_label])
        if key not in labels_dict:
            labels_dict[key] = [idx]
        else:
            labels_dict[key].append(idx)

    # for each label (multi-hot encoding), their length
    labels_dict_len = []
    labels_dict_key = []
    for key in labels_dict.keys():
        labels_dict_len.append(len(labels_dict[key]))
        labels_dict_key.append(key)
    labels_dict_len = np.array(labels_dict_len)
    labels_dict_key = np.array(labels_dict_key)

    threshold = 0
    if threshold_percentile > 0:
        threshold = np.percentile(labels_dict_len, threshold_percentile)

    labels_dict_len_threshold = labels_dict_len[labels_dict_len>0]
    labels_dict_key_threshold = labels_dict_key[labels_dict_len>0]

    # old label (multi-hot encoding) => new label (number)
    new_index_dict = {}
    for i in range(len(labels_dict_key_threshold)):
        new_index_dict[labels_dict_key_threshold[i]] = i

    # change label to new label
    Data_labels_new = np.zeros((len(Data_labels),), dtype = int)
    special_class = -1
    for key, vals in labels_dict.items():
        for val in vals:
            if key not in labels_dict_key_threshold:
                Data_labels_new[val] = special_class
            else:
                Data_labels_new[val] = new_index_dict[key]


    # distribution of classes in the dataset 
    label_to_count = {}
    for label in Data_labels_new:
        if label in label_to_count:
            label_to_count[label] += 1
        else:
            label_to_count[label] = 1

    # weight for each sample
    sample_weights = [1.0 / label_to_count[label]
               for label in Data_labels_new]

    return sample_weights, Data_labels_new, label_to_count, new_index_dict

def proportional_weight(Data_labels, class_idx):
    class_weights = np.sum(Data_labels, axis=0)[class_idx]
    return class_weights/np.sum(class_weights)

def inverse_weight(Data_labels, class_idx, K=1):
    # inv_class_weights = 1.0/np.sum(Data_labels, axis=0)[class_idx]
    # inv_class_weights = inv_class_weights / np.sum(inv_class_weights)

    # return inv_class_weights

    Data_labels = Data_labels[:,class_idx]
    N_labels = np.sum(Data_labels, axis=0).flatten()+1

    return np.log((np.sum(N_labels)-N_labels)/N_labels + K)

def inverse_weight_no_log(Data_labels, class_idx, K=1):
    # inv_class_weights = 1.0/np.sum(Data_labels, axis=0)[class_idx]
    # inv_class_weights = inv_class_weights / np.sum(inv_class_weights)

    # return inv_class_weights

    Data_labels = Data_labels[:,class_idx]
    N_labels = np.sum(Data_labels, axis=0).flatten()+1

    return (np.sum(N_labels)-N_labels)/N_labels + K
