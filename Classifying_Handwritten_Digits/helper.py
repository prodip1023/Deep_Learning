import numpy as np
from utils.utils import sigmoid,int_to_onehot


num_epochs = 50
minibatch_size = 100

def minibatch_generator(X,y,minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]


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