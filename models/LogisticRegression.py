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
            #if self.sigmoid(np.array(X.iloc[i])) != 1 and self.sigmoid(np.array(X.iloc[i])) != 0:
            z = np.dot(np.array(X.iloc[i]), self.w.T)
            J += y[i] * np.log1p(np.exp(-z)) + (1 - y[i]) * np.log1p(np.exp(z))
            #print(z)

        return [J / len(X.index)]

    def gradient_cross_entropy(self, X, y):
        dw = 0
        lam = 1e-3
        for i in range(len(X.index)):
            dw += (np.array(X.iloc[i]) * (y[i] - self.sigmoid(np.array(X.iloc[i]))))
        dw += (lam * self.w)[0]
        return dw

    def fit(self, X, y, rate, iteration):

        cost = [1]
        costList = []
        difference = 1
        min_difference = 0.05

        for iterator in range(0, iteration, 1):
        #while difference >= min_difference:
            gradient = self.gradient_cross_entropy(X, y)

            wk = self.w + rate * gradient
            difference = np.sum(list(self.w - wk))
            self.w = wk

            cost = self.cross_entropy(X, y)
            costList.append(cost)

        return costList

    # predict the data points
    def predict(self, X):
        prediction = []
        for i in range(len(X.index)):
            y_hat = self.sigmoid(np.array(X.iloc[i]))
            # if y_hat == 1:
            #     prediction.append(1)
            # else:
            prediction.append(y_hat)
                #prediction.append(np.log(y_hat / (1 - y_hat)))
        print(prediction)
        prediction = [0 if p < 0.5 else 1 for p in prediction]

        return prediction

    def evaluate_acc(self, data_y, prediction_y):
        differences = 0
        for i in range(len(prediction_y)):
            delta = np.abs(data_y.iloc[i].values[0] - prediction_y[i])
            differences = differences + delta

        return differences / len(prediction_y)

    def confusion_matrix(self, data_y, prediction_y, category):
        matrix = np.zeros(shape=(2, 2))

        for i in range(len(prediction_y)):
            if data_y.iloc[i].values[0] == category[1]:
                if prediction_y[i] == category[1]:
                    matrix[0][0] += 1  # True Positive
                else:
                    matrix[0][1] += 1  # False Positive
            elif data_y.iloc[i].values[0] == category[0]:
                if prediction_y[i] == category[0]:
                    matrix[1][1] += 1  # True Negative
                else:
                    matrix[1][0] += 1  # False Negative

        return matrix
