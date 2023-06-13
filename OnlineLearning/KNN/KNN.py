import numpy as np


class KNN:
    def __init__(self, X_train, Y_train, K):
        self.X_train = X_train
        self.Y_train = Y_train
        self.K = K

    def predict(self, X):
        y_pred = np.array([])
        for each in X:
            ed = np.sum((each - self.X_train) ** 2, axis=1)
            y_ed = np.concatenate((self.Y_train.reshape(self.Y_train.shape[0], 1), ed.reshape(ed.shape[0], 1)), axis=1)
            y_ed = y_ed[y_ed[:, 1].argsort()]
            K_neighbours = y_ed[0:self.K]
            (values, counts) = np.unique(K_neighbours[:, 0].astype(int), return_counts=True)
            y_pred = np.append(y_pred, values[np.argmax(counts)])
        return y_pred
