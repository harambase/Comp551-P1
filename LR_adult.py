from LogisticRegression import *
from Dataset2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def train(df, tdf, model, k, rate, iteration):
    # accuracies = np.ones(k)
    costList = []
    prediction_data = tdf
    X = df.iloc[:, 0:103]
    Y = np.array(df.iloc[:, 103:104])
    costList = np.append(costList, model.fit(X, Y, rate=rate, iteration=iteration))
    prediction = model.predict(prediction_data.iloc[:, 0:103])
    accuracies = model.evaluate_acc(prediction_data.iloc[:, 103:104], prediction)
    return np.mean(accuracies), costList


# Read Data
df = PrepareData.dataframe
tdf = PrepareData.testdata

# Logistic regression
lr = LogisticRegression(np.zeros((1, 103), float))
rate = .0005
iteration = 100

start_time = time.time()

result = train(df, tdf, lr, 5, rate, iteration)

end_time = time.time()
print("Elapsed time for Logistic Regression on the adult set is %g seconds" % (end_time - start_time))
print("The accuracy is %g" % result[0])

# Plot number of iterations vs Cost
plt.plot(result[1], '-y', label='a = ' + str(rate))
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Number of Iterations vs Cost")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
