from models.LogisticRegression import *
from data.Project1_data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def create_df_ionosphere(file):

    data = split_data(data_inosphere("../data/ionosphere.data"))

    input_array = pd.DataFrame(data[0]).astype(float)
    output_array = pd.DataFrame({'target': data[1]}).astype(float)
    df = pd.concat([input_array, output_array], axis=1)

    tinput_array = pd.DataFrame(data[2]).astype(float)
    toutput_array = pd.DataFrame({'target': data[3]}).astype(float)
    tdf = pd.concat([tinput_array, toutput_array], axis=1)

    #print(df)

    return df, tdf


def create_df_haberman(file):
    dataset = pd.DataFrame(pd.read_csv(file, header=None))
    return dataset


def create_df_heart(file):
    dataset = pd.DataFrame(pd.read_csv(file, header=None, delim_whitespace=True))
    return dataset


def create_df_adult(file, train):
    global mean, max, min
    data = data_adult(file, True)
    input_array = pd.DataFrame(data[0]).astype(float)

    input_array = input_array.drop(input_array.loc[:, 1:1], axis=1)
    # input_array = input_array.drop(input_array.loc[:, 103:104], axis=1)

    if train:
        mean = input_array.mean()
        min = input_array.min()

    input_array = 1e-4 * input_array
    output_array = pd.DataFrame({'105': data[1]}).astype(float)
    df = pd.concat([input_array, output_array], axis=1)
    print(df)
    return df


def k_fold(data, model, k, rate, iteration, category):
    costList = []
    df = data[0]
    tdf = data[1]
    lens = len(df.columns)

    df = df.iloc[np.random.permutation(len(df))]
    data_split = np.array_split(df, k)
    accuracies = np.ones(k)

    for i in range(0, k - 1, 1):
        df = pd.concat([df, data_split[i]]).drop_duplicates(keep=False)
        vdf = data_split[i]

        X = df.iloc[:, 0:lens - 1]
        Y = df.iloc[:, lens - 1:lens]

        pX = vdf.iloc[:, 0:lens - 1]
        pY = vdf.iloc[:, lens - 1:lens]
        costList = np.append(costList, model.fit(X, np.array(Y), rate=rate, iteration=iteration))
        prediction = model.predict(pX)
        accuracies[i] = model.evaluate_acc(pY, prediction)
        #matrix = model.confusion_matrix(pY, prediction, category)
        #print(matrix)

    prediction = model.predict(tdf.iloc[:, 0:lens - 1])
    print(model.evaluate_acc(tdf.iloc[:, lens - 1:lens], prediction))
    matrix = model.confusion_matrix(tdf.iloc[:, lens - 1:lens], prediction, category)
    print(matrix)

    return np.mean(accuracies), costList


def train_and_predict_ionosphere():
    # Read Data
    data = create_df_ionosphere("../data/ionosphere.data")
    train_and_predict_by_k_fold(data, 'ionosphere', [0, 1])


def train_and_predict_haberman():
    # Read Data
    df = create_df_ionosphere("../data/haberman.data")
    train_and_predict_by_k_fold(df, 'haberman', [1, 2])


def train_and_predict_heart():
    # Read Data
    df = create_df_heart("../data/heart.data")
    train_and_predict_by_k_fold(df, 'heart', [1, 2])


def train_and_predict_by_k_fold(df, name, category):
    # Logistic regression
    rate = .005
    iteration = 50
    lr = LogisticRegression(np.zeros((1, len(df[0].columns) - 1), float))

    start_time = time.time()

    result = k_fold(df, lr, 5, rate, iteration, category)

    end_time = time.time()

    print("Elapsed time for Logistic Regression on the " + name + " set is %g seconds" % (end_time - start_time))
    print("The accuracy is %g" % result[0])

    # Plot number of iterations vs Cost
    plt.plot(result[1], '-y', label='a = ' + str(rate))
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Number of Iterations vs Cost")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def train_and_predict_adult(size):
    # Read Data
    df = create_df_adult("../data/adult.data", True)
    tdf = create_df_adult("../data/adult.test", False)
    lens = len(df.columns)
    # Logistic regression
    lr = LogisticRegression(np.zeros((1, lens - 1), float))
    rate = .01
    iteration = 20
    costList = []
    start_time = time.time()

    # df = df.iloc[np.random.permutation(len(df))]

    X = df.iloc[0:size, 0:lens - 1]
    Y = df.iloc[0:size, lens - 1:lens]

    pX = tdf.iloc[:, 0:lens - 1]
    pY = tdf.iloc[:, lens - 1:lens]

    costList = np.append(costList, lr.fit(X, np.array(Y), rate=rate, iteration=iteration))
    prediction = lr.predict(pX)
    print(costList)
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


train_and_predict_ionosphere()
# train_and_predict_heart()
# train_and_predict_haberman()
# size = [200, 1000, 5000, 10000, 20000, 30161]
# for i in range(0, 5, 1):
#     train_and_predict_adult(size[i])
