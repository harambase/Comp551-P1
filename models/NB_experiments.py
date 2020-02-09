import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from Naive_Bayes import NaiveBayes
from Project1_data import *

def k_fold_validation_acc(class_labels,k,X_train,y_train,ld):
	kf = KFold(n_splits=k,shuffle=True,random_state=10)
	cv_acc=0
	NB=NaiveBayes(class_labels)
	for train, test in kf.split(X_train):
		NB.fit(X_train[train],y_train[train],ld)
		y1=NB.predict(X_train[test],ld)
		cv_acc=cv_acc+NB.evaluate_acc(y_train[test],y1)
	cv_acc=cv_acc/k
	print("cross validation accuracy=:",cv_acc)
	return cv_acc

# adult
X_train,y_train=data_adult("adult.data",whether_One_hot_coding=False)
X_test, y_test = data_adult("adult.test",whether_One_hot_coding=False)
ld=[-1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,-1,0.1]
print("adult:")
cv_acc_adult=k_fold_validation_acc([0,1],5,X_train,y_train,ld)

#ionosphere
X,y=data_inosphere("ionosphere.data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ld=[-1]*34
print("ionosphere:")
cv_acc_ionosphere=k_fold_validation_acc([0,1],5,X_train,y_train,ld)

#haberman
X,y=data_haberman("haberman.data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ld=[1,-1,-1]
print("haberman:")
cv_acc_haberman=k_fold_validation_acc([1,2],5,X_train,y_train,ld)

#heart
X,y=data_heart("heart.data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ld=[-1,0.1,0.1,-1,-1,0.1,0.1,-1,0.1,-1,0.1,0.1,0.1]
print("heart:")
cv_acc_heart=k_fold_validation_acc([1,2],5,X_train,y_train,ld)