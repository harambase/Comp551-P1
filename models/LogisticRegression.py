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
            z = np.dot(np.array(X.iloc[i]), self.w.T)
            J += y[i] * np.log1p(np.exp(-z)) + (1 - y[i]) * np.log1p(np.exp(z))
        return [J / len(X.index)]

    def gradient_cross_entropy(self, X, y, lam):
        dw = 0
        for i in range(len(X.index)):
            dw += (np.array(X.iloc[i]) * (y[i] - self.sigmoid(np.array(X.iloc[i]))))
        dw -= (lam * self.w)[0]
        return dw

    def fit(self, X, y, rate, lam, iteration):
        cost = []
        difference = np.inf
        min_difference = 1e-2
        eps = 0.5
        beta = .99
        gradient = np.inf
        dw = 0
        k = 0
        for iterator in range(0, iteration, 1):
        #while difference >= min_difference:
        #while np.linalg.norm(gradient) > eps:
            gradient = self.gradient_cross_entropy(X, y, lam)
            dw = (1 - beta) * gradient + beta * dw #M
            wk = self.w + rate * dw
            difference = np.sum(list(self.w - wk))
            self.w = wk
            k += 1
            #print(np.linalg.norm(gradient))

            cost.append(self.cross_entropy(X, y))

        return cost

    # predict the data points
    def predict(self, X, category):
        prediction = []
        for i in range(len(X.index)):
            y_hat = self.sigmoid(np.array(X.iloc[i]))
            if y_hat == 1:
                prediction.append(1)
            else:
                prediction.append(np.log(y_hat / (1 - y_hat)))

        prediction = [category[0] if p < 0 else category[1] for p in prediction]

        return prediction

    def evaluate_acc(self, data_y, prediction_y):
        acc = 0
        for i in range(len(prediction_y)):
            if data_y.iloc[i].values[0] == prediction_y[i]:
                acc += 1

        return acc / len(prediction_y)

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
