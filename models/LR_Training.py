from models.LogisticRegression import *
from data.Project1_data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

def k_fold(data, k, rates, lambs, iteration, category):
    costList = []
    df = data[0]
    tdf = data[1]
    lens = len(df.columns)
    data_split = np.array_split(df, k)
    best_r = np.inf
    best_l = np.inf
    max_acc = -np.inf

    for l in range(len(lambs)):
        lam = lambs[l]
        #print("lam=%f" %lam)
        for r in range(len(rates)):
            rate = rates[r]
            accuracy = 0
            #print("rate=%f" %rate)
            for i in range(0, k, 1):
                dfk = pd.concat([df, data_split[i]]).drop_duplicates(keep=False)
                vdfk = data_split[i]

                X = dfk.iloc[:, 0:lens - 1]
                Y = dfk.iloc[:, lens - 1:lens]

                pX = vdfk.iloc[:, 0:lens - 1]
                pY = vdfk.iloc[:, lens - 1:lens]

                nX, nPx = data_normalized(np.array(X), np.array(pX))
                nX = pd.DataFrame(nX).astype(float)
                nPx = pd.DataFrame(nPx).astype(float)
                # print(nX)
                model = LogisticRegression(np.zeros((1, len(nX.columns)), float))
                costList = np.append(costList, model.fit(nX, np.array(Y), rate, lam, iteration))
                prediction = model.predict(nPx, category)
                accuracy += model.evaluate_acc(pY, prediction)

            mean_acc = accuracy / k
            print(mean_acc)
            if mean_acc > max_acc:
                max_acc = mean_acc
                best_r = r
                best_l = l

    X = df.iloc[:, 0:lens - 1]
    Y = df.iloc[:, lens - 1:lens]

    pX = tdf.iloc[:, 0:lens - 1]
    pY = tdf.iloc[:, lens - 1:lens]

    nX, nPx = data_normalized(np.array(X), np.array(pX))
    nX = pd.DataFrame(nX).astype(float)
    nPx = pd.DataFrame(nPx).astype(float)
    print(rates[best_r])
    print(lambs[best_l])

    model = LogisticRegression(np.zeros((1, len(nX.columns)), float))
    costList = np.append(costList, model.fit(nX, np.array(Y), rates[best_r], lambs[best_l], iteration))
    prediction = model.predict(nPx, category)
    acc = model.evaluate_acc(pY, prediction)
    matrix = model.confusion_matrix(pY, prediction, category)
    print(matrix)

    return acc, costList

def k_fold_plot(data, k, rate, lambs, iteration, category):
    costList = []
    df = data[0]
    tdf = data[1]
    lens = len(df.columns)
    data_split = np.array_split(df, k)
    best_l = np.inf
    max_acc = -np.inf

    for l in range(len(lambs)):
        lam = lambs[l]
        # print("lam=%f" %lam)
        accuracy = 0
        # print("rate=%f" %rate)
        for i in range(0, k, 1):
            dfk = pd.concat([df, data_split[i]]).drop_duplicates(keep=False)
            vdfk = data_split[i]

            X = dfk.iloc[:, 0:lens - 1]
            Y = dfk.iloc[:, lens - 1:lens]

            pX = vdfk.iloc[:, 0:lens - 1]
            pY = vdfk.iloc[:, lens - 1:lens]

            nX, nPx = data_normalized(np.array(X), np.array(pX))
            nX = pd.DataFrame(nX).astype(float)
            nPx = pd.DataFrame(nPx).astype(float)

            model = LogisticRegression(np.zeros((1, len(nX.columns)), float))
            costList = np.append(costList, model.fit(nX, np.array(Y), rate, lam, iteration))
            prediction = model.predict(nPx, category)
            accuracy += model.evaluate_acc(pY, prediction)

        mean_acc = accuracy / k
        if mean_acc > max_acc:
            max_acc = mean_acc
            best_l = l

    X = df.iloc[:, 0:lens - 1]
    Y = df.iloc[:, lens - 1:lens]

    pX = tdf.iloc[:, 0:lens - 1]
    pY = tdf.iloc[:, lens - 1:lens]

    nX, nPx = data_normalized(np.array(X), np.array(pX))
    nX = pd.DataFrame(nX).astype(float)
    nPx = pd.DataFrame(nPx).astype(float)
    # print(rates[best_r])
    # print(lambs[best_l])

    model = LogisticRegression(np.zeros((1, len(nX.columns)), float))
    costList = np.append(costList, model.fit(nX, np.array(Y), rate, lambs[best_l], iteration))
    prediction = model.predict(nPx, category)
    acc = model.evaluate_acc(pY, prediction)
    matrix = model.confusion_matrix(pY, prediction, category)
    print(matrix)

    return acc, costList

def split_data(data):
    return split_into_train_test(data[0], data[1])


def train_and_predict_ionosphere():
    # Read Data
    data = split_data(data_ionosphere("../data/ionosphere.data"))
    train_and_predict_by_k_fold(data, 'ionosphere', [0, 1])

def train_and_predict_ionosphere_plot():
    # Read Data
    data = split_data(data_ionosphere("../data/ionosphere.data"))
    train_and_predict_by_k_fold_plot(data, 'ionosphere', [0, 1])

def train_and_predict_haberman():
    # Read Data
    data = split_data(data_haberman("../data/haberman.data"))
    train_and_predict_by_k_fold(data, 'haberman', [1, 2])


def train_and_predict_heart():
    # Read Data
    data = split_data(data_heart("../data/heart.data", True))
    train_and_predict_by_k_fold(data, 'heart', [1, 2])


def procedure(data, rate, lambs, iteration, category, resultQ):
    print(iteration)
    input_array = pd.DataFrame(data[0]).astype(float)
    output_array = pd.DataFrame({'target': data[1]}).astype(float)
    df = pd.concat([input_array, output_array], axis=1)

    tinput_array = pd.DataFrame(data[2]).astype(float)
    toutput_array = pd.DataFrame({'target': data[3]}).astype(float)
    tdf = pd.concat([tinput_array, toutput_array], axis=1)

    result = k_fold_plot([df, tdf], 5, rate, lambs, iteration, category)
    resultQ.put(result)


def train_and_predict_by_k_fold_plot(data, name, category):
    # Logistic regression
    rates = [0.5, 0.4, 0.1, 0.05, 0.01]
    #rates = [0.5, 0.4]
    lambs = [0.1, 0.5, 1, 5, 10]
    accuracies = []
    max_iter = 100

    for r in range(0, len(rates)):
        rate = rates[r]
        accuracies = []
        for iteration in range(0, max_iter, 5):
            input_array = pd.DataFrame(data[0]).astype(float)
            output_array = pd.DataFrame({'target': data[1]}).astype(float)
            df = pd.concat([input_array, output_array], axis=1)

            tinput_array = pd.DataFrame(data[2]).astype(float)
            toutput_array = pd.DataFrame({'target': data[3]}).astype(float)
            tdf = pd.concat([tinput_array, toutput_array], axis=1)

            start_time = time.time()
            result = k_fold_plot([df, tdf], 5, rate, lambs, iteration, category)
            end_time = time.time()
            accuracies.append(result[0])

            print("iteration is %g" % iteration)
            print("accuracy is %g" % result[0])
            print("Elapsed time for Logistic Regression on the " + name + " set is %g seconds" % (end_time - start_time))

        print(accuracies)
        plt.plot(range(0, max_iter, 5), accuracies, marker='o', linestyle='dashed', label='a_' + str(r) + '= ' + str(rate))

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Number of Iterations vs Accuracy")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def train_and_predict_by_k_fold(data, name, category):
    # Logistic regression
    rates = [0.5, 0.4, 0.1, 0.05, 0.01]
    lambs = [0.1, 0.5, 1, 5, 10]
    accuracies = []

    iteration = 20
    input_array = pd.DataFrame(data[0]).astype(float)
    output_array = pd.DataFrame({'target': data[1]}).astype(float)
    df = pd.concat([input_array, output_array], axis=1)

    tinput_array = pd.DataFrame(data[2]).astype(float)
    toutput_array = pd.DataFrame({'target': data[3]}).astype(float)
    tdf = pd.concat([tinput_array, toutput_array], axis=1)

    start_time = time.time()
    result = k_fold([df, tdf], 5, rates, lambs, iteration, category)
    end_time = time.time()
    accuracies.append(result[0])

    print("iteration is %g" % iteration)
    print("accuracy is %g" % result[0])
    print("Elapsed time for Logistic Regression on the " + name + " set is %g seconds" % (end_time - start_time))


    # Plot number of iterations vs Cost
    plt.plot(accuracies, '-y', label='a = ' + str(rates))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Number of Iterations vs Accuracy")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def train_and_predict_adult(size):
    # Read Data
    data = data_adult("../data/adult.data", True)
    tdata = data_adult("../data/adult.test", True)
    train_and_predict_by_k_fold([data[0][:size, :], data[1][:size], tdata[0], tdata[1]], 'adult', [0, 1])



if __name__ == '__main__':
    data = split_data(data_ionosphere("../data/ionosphere.data"))
    category = [0, 1]
    # Logistic regression
    rates = [0.5, 0.4, 0.1, 0.05, 0.01]
    #rates = [0.5, 0.4]
    lambs = [0.1, 0.5, 1, 5, 10]
    accuracies = []
    max_iter = 100

    for r in range(0, len(rates)):
        rate = rates[r]
        accuracies = []
        jobs = []
        resultQ = mp.Queue()

        for iteration in range(0, max_iter, 5):
            p = mp.Process(target=procedure, args=(data, rate, lambs, iteration, category, resultQ))
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        for a in range(resultQ.qsize()):
            accuracies.append(resultQ.get()[0])
        print(accuracies)
        plt.plot(range(0, max_iter, 5), accuracies, marker='o', linestyle='dashed',
                 label='a_' + str(r) + '= ' + str(rate))

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Number of Iterations vs Accuracy")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


#train_and_predict_ionosphere()
#train_and_predict_ionosphere_plot()
# train_and_predict_heart()
# train_and_predict_haberman()
# size = [200, 1000, 5000, 10000, 20000, 30161]
# for i in range(len(size)):
#     print("Size=%d" % size[i])
#     train_and_predict_adult(size[i])
#train_and_predict_adult(20000)
