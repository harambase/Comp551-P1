import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from Naive_Bayes import NaiveBayes
from Project1_data import *


# adult
X_train,y_train=data_adult("adult.data",whether_One_hot_coding=False)
X_test, y_test = data_adult("adult.test",whether_One_hot_coding=False)
a=0.2
ld=[-1,a,-1,a,-1,a,a,a,a,a,-1,-1,-1,a]

NB=NaiveBayes([0,1])
NB.fit(X_train,y_train,ld)
y1=NB.predict(X_test,ld)
acc=NB.evaluate_acc(y_test,y1)
print("adult:",acc)

#ionosphere
X,y=data_ionosphere("ionosphere.data")
X_train, y_train, X_test,y_test = split_into_train_test(X,y)
ld=[-1]*34
print(len(y_train))

NB=NaiveBayes([0,1])
NB.fit(X_train,y_train,ld)
y1=NB.predict(X_test,ld)
acc=NB.evaluate_acc(y_test,y1)
print("ionosphere:",acc)

#haberman
X,y=data_haberman("haberman.data")
X_train, y_train, X_test,y_test = split_into_train_test(X,y)
ld=[1,-1,-1]

NB=NaiveBayes([1,2])
NB.fit(X_train,y_train,ld)
y1=NB.predict(X_test,ld)
acc=NB.evaluate_acc(y_test,y1)
print("haberman:",acc)

#heart
X,y=data_heart("heart.data",whether_One_hot_coding=False)
X_train, y_train, X_test,y_test = split_into_train_test(X,y)
a=1
ld=[-1,a,a,-1,-1,a,a,-1,a,-1,a,-1,a]

NB=NaiveBayes([1,2])
NB.fit(X_train,y_train,ld)
y1=NB.predict(X_test,ld)
acc=NB.evaluate_acc(y_test,y1)
print("heart:",acc)