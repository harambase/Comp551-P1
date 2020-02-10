import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from Naive_Bayes import NaiveBayes
from Project1_data import *
import matplotlib.pyplot as plt

def k_fold_validation_acc(class_labels,k,X_train,y_train,ld):
	cv_acc=0
	NB=NaiveBayes(class_labels)
	l=np.arange(len(y_train))

	kf=np.array_split(l,k)

	for i in range(k):
		train=np.array([])
		test=kf[i]
		for j in range(k):
			if j is not i:
				train=np.concatenate((train, kf[j]), axis=None)

		test = test.astype(int)
		train = train.astype(int)

		NB.fit(X_train[train],y_train[train],ld)
		y1=NB.predict(X_train[test],ld)
		cv_acc=cv_acc+NB.evaluate_acc(y_train[test],y1)

	cv_acc=cv_acc/k
	return cv_acc

a_range=np.arange(0.1,2,0.1)

# adult
X_train,y_train=data_adult("adult.data",whether_One_hot_coding=False)
cv_acc_adult=[]
for a in a_range:
	ld=[-1,a,-1,a,-1,a,a,a,a,a,-1,-1,-1,a]
	cv_acc_adult.append(k_fold_validation_acc([0,1],5,X_train,y_train,ld))




# # # ionosphere
# # X,y=data_inosphere("ionosphere.data")
# # X_train, y_train, X_test,y_test = split_into_train_test(X,y)
# # cv_acc_ionosphere=[]
# # for a in a_range:
# # 	ld=[a,a]+[-1]*32
# # 	cv_acc_ionosphere.append(k_fold_validation_acc([0,1],5,X_train,y_train,ld))


# #haberman
# X,y=data_haberman("haberman.data")
# X_train,  y_train, X_test,y_test = split_into_train_test(X,y)
# cv_acc_haberman=0
# # for a in a_range:
# ld=[-1,-1,-1]
# 	# cv_acc_haberman.append(k_fold_validation_acc([1,2],5,X_train,y_train,ld))
# print(k_fold_validation_acc([1,2],5,X_train,y_train,ld))


# heart
X,y=data_heart("heart.data",whether_One_hot_coding=False)
X_train,  y_train, X_test,y_test = split_into_train_test(X,y)
cv_acc_heart=[]
for a in a_range:
	ld=[-1,a,a,-1,-1,a,a,-1,a,-1,a,-1,a]
	cv_acc_heart.append(k_fold_validation_acc([1,2],5,X_train,y_train,ld))

# cv_acc_adult=[0.8272994441271504, 0.8272994441271504, 0.8272662876284766, 0.8272331366256589, 0.8272994441271504, 0.8272994441271504, 0.8272662931243324, 0.8272331366256587, 0.8271999801269849, 0.8271336726254935, 0.8271668236283114, 0.8271336726254935, 0.8271336726254935, 0.8271336781213494, 0.8271005216226757, 0.8271005216226757, 0.8270673651240019, 0.8271005216226757, 0.8270673706198577]
# cv_acc_heart=[0.8391304347826086, 0.8434782608695652, 0.8434782608695652, 0.8434782608695652, 0.8434782608695652, 0.8434782608695652, 0.8478260869565217, 0.8478260869565217, 0.8478260869565217, 0.8478260869565217, 0.8478260869565217, 0.8478260869565217, 0.8434782608695652, 0.8434782608695652, 0.8434782608695652, 0.8434782608695652, 0.8434782608695652, 0.8434782608695652, 0.8434782608695652]
print("adult:",cv_acc_adult)
# print("ionosphere:",cv_acc_ionosphere)
# print("haberman:",cv_acc_haberman)
print("heart:",cv_acc_heart)
plt.plot(a_range,cv_acc_adult, color="k", linestyle="-", linewidth=1,label = 'Adult')
# plt.plot(a_range,cv_acc_ionosphere, color="g", linestyle="--", linewidth=1,label ='Ionosphere')
# plt.plot(a_range,cv_acc_haberman, color="r", linestyle="--", linewidth=1,label ='Haberman')

plt.plot(a_range,cv_acc_heart, color="b", linestyle="-.",  linewidth=1,label ='Heart')
plt.xlabel("Smoothing parameter",fontsize = 11)
plt.ylabel("Cross validation accuracy")
plt.title("Smoothing parameter v.s. Cross validation accuracy")
plt.legend(loc='center right')
plt.show()
