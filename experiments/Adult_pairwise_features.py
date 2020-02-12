import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from Naive_Bayes import NaiveBayes
from Project1_data import *
import matplotlib.pyplot as plt


# adult
X_train,y_train=data_adult("adult.data",whether_One_hot_coding=False)

age=X_train[:,0]
education_num=X_train[:,4]
capital_gain=X_train[:,10]
capital_loss=X_train[:,11]
hours_per_week=X_train[:,12]

age=list(map(int, age))
education_num=list(map(int, education_num))
capital_gain=list(map(int, capital_gain))
capital_loss=list(map(int, capital_loss))
hours_per_week=list(map(int, hours_per_week))
age_0=[]
age_1=[]
education_num_0=[]
education_num_1=[]
capital_gain_0=[]
capital_gain_1=[]
capital_loss_0=[]
capital_loss_1=[]
hours_per_week_0=[]
hours_per_week_1=[]

for i in range(len(age)):
	if y_train[i]==0:
		age_0.append(age[i])
		education_num_0.append(education_num[i])
		capital_gain_0.append(capital_gain[i])
		capital_loss_0.append(capital_loss[i])
		hours_per_week_0.append(hours_per_week[i])
	else:
		age_1.append(age[i])
		education_num_1.append(education_num[i])
		capital_gain_1.append(capital_gain[i])
		capital_loss_1.append(capital_loss[i])
		hours_per_week_1.append(hours_per_week[i])

#education number v.s. age	
plt.figure()	
plt.scatter(age_0,education_num_0,color="k", marker='.',label = '<=50k')
plt.scatter(age_1,education_num_1,color="r", marker='1',label = '>50k')
plt.xlabel("age",fontsize = 11)
plt.ylabel("education number")
plt.legend(loc='lower right')
plt.show()

#capital gain v.s. capital loss		
plt.figure(num=2)	
plt.scatter(capital_gain_0,capital_loss_0,color="k", marker='.',label = '<=50k')
plt.scatter(capital_gain_1,capital_loss_1,color="r", marker='1',label = '>50k')
plt.xlabel("capital gain",fontsize = 11)
plt.ylabel("capital loss")
plt.legend(loc='upper right')
plt.show()

#hours per week v.s. age	
plt.figure(num=3)		
plt.scatter(age_0,hours_per_week_0,color="k", marker='.',label = '<=50k')
plt.scatter(age_1,hours_per_week_1,color="r", marker='1',label = '>50k')
plt.xlabel("age",fontsize = 11)
plt.ylabel("hours per week")
plt.legend(loc='upper right')
plt.show()

#hours per week v.s. education number		
plt.figure(num=4)	
plt.scatter(hours_per_week_0,education_num_0,color="k", marker='.',label = '<=50k')
plt.scatter(hours_per_week_1,education_num_1,color="r", marker='1',label = '>50k')
plt.xlabel("hours per week",fontsize = 11)
plt.ylabel("education number")
plt.legend(loc='lower right')
plt.show()
