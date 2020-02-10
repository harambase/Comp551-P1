import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from Naive_Bayes import NaiveBayes
from Project1_data import *


# adult
X_train,y_train=data_adult("adult.data",whether_One_hot_coding=False)
X_test, y_test = data_adult("adult.test",whether_One_hot_coding=False)
a=0.1
ld=[-1,a,-1,a,-1,a,a,a,a,a,-1,-1,-1,a]

size_ = [200, 1000, 5000, 10000, 20000, 30161]

acc=[]

for s in size_:
	X_train_sub=X_train[0:s,:]
	y_train_sub=y_train[0:s]
	NB=NaiveBayes([0,1])
	NB.fit(X_train_sub,y_train_sub,ld)
	y1=NB.predict(X_test,ld)
	acc.append(NB.evaluate_acc(y_test,y1))

plt.plot(size_,acc, color="k", linestyle="-",  linewidth=1,marker='.')
plt.yticks(np.arange(0.4,1.1,0.2))
plt.xlabel("Traing set size",fontsize = 11)
plt.ylabel("Test accuracy")
plt.title("Traing set size v.s.Test accuracy")
plt.show()