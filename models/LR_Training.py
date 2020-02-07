from models.LogisticRegression import *
from data.Project1_data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def create_df_ionosphere(file):
    pd.set_option('display.max_columns', 35)

    dataset = pd.DataFrame(pd.read_csv(file, header=None))

    dataset.replace('g', 1, inplace=True)
    dataset.replace('b', 0, inplace=True)

    return dataset


def create_df_haberman(file):
    pd.set_option('display.max_columns', 35)

    dataset = pd.DataFrame(pd.read_csv(file, header=None))

    return dataset


def create_df_adult(file):
    data = data_adult(file, True)
    input_array = pd.DataFrame(data[0]).astype(float)

    input_array = input_array.drop(input_array.loc[:, 1:3], axis=1)
    input_array = input_array.drop(input_array.loc[:, 103:104], axis=1)
    input_array = (input_array - input_array.min()) / (input_array.max() - input_array.min())
    output_array = pd.DataFrame({'105': data[1]}).astype(float)
    df = pd.concat([input_array, output_array], axis=1)
    print(df)
    return df


def k_fold(df, model, k, rate, iteration, category):
    costList = []
    lens = len(df.columns)

    df = df.iloc[np.random.permutation(len(df))]
    data_split = np.array_split(df, k)
    accuracies = np.ones(k)

    df = pd.concat([df, data_split[1]]).drop_duplicates(keep=False)
    tdf = data_split[1]

    X = df.iloc[:, 0:lens - 1]
    Y = df.iloc[:, lens - 1:lens]

    pX = tdf.iloc[:, 0:lens - 1]
    pY = tdf.iloc[:, lens - 1:lens]

    for i in range(0, k, 1):
        costList = np.append(costList, model.fit(X, np.array(Y), rate=rate, iteration=iteration))
        prediction = model.predict(pX)
        accuracies[i] = model.evaluate_acc(pY, prediction)
        matrix = model.confusion_matrix(pY, prediction, category)
        print(matrix)

    return np.mean(accuracies), costList


def train_and_predict_ionosphere():
    # Read Data
    df = create_df_ionosphere("../data/ionosphere.data")

    # Logistic regression
    lr = LogisticRegression(np.zeros((1, 34), float))
    rate = .0005
    iteration = 10

    start_time = time.time()

    result = k_fold(df, lr, 5, rate, iteration, [0, 1])

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


def train_and_predict_haberman():
    # Read Data
    df = create_df_ionosphere("../data/haberman.data")

    # Logistic regression
    lr = LogisticRegression(np.zeros((1, len(df.columns)-1), float))
    rate = .0005
    iteration = 50

    start_time = time.time()

    result = k_fold(df, lr, 5, rate, iteration, [1, 2])

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


def train_and_predict_adult():
    # Read Data
    df = create_df_adult("../data/adult.data")
    tdf = create_df_adult("../data/adult.test")
    lens = len(df.columns)
    # Logistic regression
    lr = LogisticRegression(np.zeros((1, lens - 1), float))
    rate = .0005
    iteration = 10

    start_time = time.time()

    df = df.iloc[np.random.permutation(len(df))]

    costList = []

    X = df.iloc[:, 0:lens - 1]
    Y = df.iloc[:, lens - 1:lens]

    pX = tdf.iloc[:, 0:lens - 1]
    pY = tdf.iloc[:, lens - 1:lens]

    costList = lr.fit(X, np.array(Y), rate=rate, iteration=iteration)
    prediction = lr.predict(pX)
    print(np.array(pY), prediction)
    accuracies = lr.evaluate_acc(pY, prediction)
    matrix = lr.confusion_matrix(pY, prediction, [0, 1])
    print(matrix)

    end_time = time.time()
    print("Elapsed time for Logistic Regression on the adult set is %g seconds" % (end_time - start_time))
    print("The accuracy is %g" % accuracies)
    print(costList)
    # Plot number of iterations vs Cost
    plt.plot(costList, '-y', label='a = ' + str(rate))
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Number of Iterations vs Cost")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


train_and_predict_haberman()
#train_and_predict_adult()
