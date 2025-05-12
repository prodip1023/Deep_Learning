import numpy as np

def sigmoid(z):
    return 1./ (1. + np.exp(-z))

def int_to_onehot(y,num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i,val in enumerate(y):
        ary[i, int(val)] = 1
    
    return ary


def mse_loss(targets,probas,num_labels=10):
    """
    Mean Squared Error loss function
    """
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas) ** 2)

def accuracy(targets,predicted_labels):
    """
    Accuracy metric
    """
    return np.mean(predicted_labels == targets)