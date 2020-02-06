import numpy as np


class LogisticRegression:

    def __init__(self, w):
        self.w = w

    def sigmoid(self, x):
        a = np.dot(self.w, x)[0]
        return 1.0 / (1 + np.exp(-a))

    def cross_entropy(self, X, y):
        J = 0
        for i in range(len(X.index)):
            if self.sigmoid(np.array(X.iloc[i])) != 1 and self.sigmoid(np.array(X.iloc[i])) != 0:
                z = np.dot(np.array(X.iloc[i]), self.w.T)
                J += y[i] * np.log1p(np.exp(-z)) + (1 - y[i]) * np.log1p(np.exp(z))

        return [J / len(X.index)]

    def gradient_cross_entropy(self, X, y):
        dw = 0
        for i in range(len(X.index)):
            dw += (np.array(X.iloc[i]) * (y[i] - self.sigmoid(np.array(X.iloc[i]))))
        return dw

    def fit(self, X, y, rate=0.001, iteration=50000):

        costList = []

        for iterator in range(0, iteration, 1):
            gradient = self.gradient_cross_entropy(X, y)

            self.w = self.w + rate * gradient
            cost = self.cross_entropy(X, y)
            costList.append(cost)

        return costList

    # predict the data points
    def predict(self, X):
        prediction = []
        for i in range(len(X.index)):
            y_hat = self.sigmoid(np.array(X.iloc[i]))
            if y_hat == 1:
                prediction.append(1)
            else:
                prediction.append(np.log(y_hat / (1 - y_hat)))

        # #softmax:
        # yh = np.exp(z)
        # yh /= np.sum(yh, 0)

        prediction = [0 if p <= 0 else 1 for p in prediction]

        return prediction

    def evaluate_acc(self, data_y, prediction_y):
        differences = 0
        for i in range(len(prediction_y)):
            differences = differences + np.abs(data_y.iloc[i].values[0] - prediction_y[i])

        print(differences)
        return differences / len(prediction_y)
