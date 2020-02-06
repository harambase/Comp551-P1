from LogisticRegression import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def create_df(file):
    pd.set_option('display.max_columns', 35)

    dataset = pd.DataFrame(pd.read_csv(file, header=None))

    dataset.replace('g', 1, inplace=True)
    dataset.replace('b', 0, inplace=True)

    return dataset


def k_fold(df, model, k, rate, iteration):
    all_data = df.iloc[np.random.permutation(len(df))]
    data_split = np.array_split(all_data, k)
    accuracies = np.ones(k)
    costList = []
    training_data = pd.concat([all_data, data_split[1]]).drop_duplicates(keep=False)
    prediction_data = data_split[1]
    X = training_data.iloc[:, 0:34]
    Y = np.array(training_data.iloc[:, 34:35])

    for i in range(0, k, 1):
        costList = np.append(costList, model.fit(X, Y, rate=rate, iteration=iteration))

        prediction = model.predict(prediction_data.iloc[:, 0:34])

        accuracies[i] = model.evaluate_acc(prediction_data.iloc[:, 34:35], prediction)

    return np.mean(accuracies), costList


# Read Data
df = create_df("ionosphere.data")

# Logistic regression
lr = LogisticRegression(np.zeros((1, 34), float))
rate = .0005
iteration = 10

start_time = time.time()

result = k_fold(df, lr, 5, rate, iteration)

end_time = time.time()
print("Elapsed time for Logistic Regression on the ionosphere set is %g seconds" % (end_time - start_time))
print("The accuracy is %g" % result[0])

# Plot number of iterations vs Cost
plt.plot(result[1], '-y', label='a = ' + str(rate))
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Number of Iterations vs Cost")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
