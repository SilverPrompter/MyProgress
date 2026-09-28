import numpy as np

probs = np.array([
    [0.7,0.1, 0.1, 0.1],
    [0.1, 0.1, 0.7, 0.1],
    [ 0.1, 0.6, 0.2, 0.1],
])
targets = np.array([0,2,1])
rows = np.arange(len(targets))


def cross_entropy(probs, targets):
    rows = np.arange(len(targets))
    correct = probs[rows, targets]
    loss = -np.mean(np.log(correct + 1e-9))
    return loss
