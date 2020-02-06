import numpy as np 
from sklearn.preprocessing import OneHotEncoder
#'/Users/tianchima/Desktop/ionosphere.data'
def data_inosphere(data_path_ionosphere):
	with open(data_path_ionosphere, 'r') as f:
		a = f.readlines()
	for i in range(len(a)):
		a[i] = a[i].split(',')
	input_array = np.empty([len(a),len(a[0])-1], dtype = float)
	output_array = np.empty([len(a)], dtype = int) 
	for i in range(len(a)):
		for j in range(len(a[0])-1):
			input_array[i][j] = float(a[i][j])
	for i in range(len(a)):
		if a[i][-1].strip() == 'g':
			# 1 for good
			output_array[i] = 1  
			# 0 for bad
		else:
			output_array[i] = 0
	return input_array, output_array

#'/Users/tianchima/Desktop/haberman.data'
def data_haberman(data_path_haberman):
	with open(data_path_haberman, 'r') as f:
		a = f.readlines()
	for i in range(len(a)):
		a[i] = a[i].split(',')
	input_array = np.empty([len(a),len(a[0])-1], dtype = float)
	output_array = np.empty([len(a)], dtype = int) 
	for i in range(len(a)):
		for j in range(len(a[0])-1):
			input_array[i][j] = float(a[i][j])
	for i in range(len(a)):
		if a[i][-1].strip() == '1':
			# 1 for the patient survived 5 years or longer
			output_array[i] = 1  
			# 2 for the patient died within 5 year
		else:
			output_array[i] = 2
	return input_array, output_array

#'/Users/tianchima/Desktop/heart.dat'
def data_heart(data_path_heart):
	with open(data_path_heart, 'r') as f:
		a = f.readlines()
	for i in range(len(a)):
		a[i] = a[i].split()
	input_array = np.empty([len(a),len(a[0])-1], dtype = float)
	output_array = np.empty([len(a)], dtype = int) 
	for i in range(len(a)):
		for j in range(len(a[0])-1):
			input_array[i][j] = float(a[i][j])
	for i in range(len(a)):
		if a[i][-1].strip() == '1':
			# 1 for absence of heart disease
			output_array[i] = 1  
			# 2 for presence of heart disease
		else:
			output_array[i] = 2
	return input_array, output_array

#'/Users/tianchima/Desktop/adult.data'
# test data = '/Users/tianchima/Desktop/adult.test'
def data_adult(data_path_adult):
	with open(data_path_adult, 'r') as f:
		a = f.readlines()
	a = a[:-1]
	for i in range(len(a)):
		a[i] = a[i].split(',')
	for i in range(len(a)):
		for j in range(len(a[i])):
			a[i][j] = a[i][j].strip()
	new_a = []
	for i in range(len(a)):
		normal = True
		for j in range(len(a[0])):
			if a[i][j] == '?':
				normal = False
				break
		if normal == True:
			new_a.append(a[i])
	list_input_continuous_features = []
	list_input_categorical_features = []
	output_array = np.empty([len(new_a)], dtype = int) 
	for i in range(len(new_a)):
		list_i1 = []
		list_i2 = []
		for j in range(len(new_a[i])-1):
			if new_a[i][j].isdigit():
				list_i1.append(new_a[i][j])
			else:
				list_i2.append(new_a[i][j])
		list_input_continuous_features.append(list_i1)
		list_input_categorical_features.append(list_i2)
		if new_a[i][-1] == '>50K':
			output_array[i] = 1
		else:
			output_array[i] = 0
	enc = OneHotEncoder()
	enc.fit(list_input_categorical_features)
	list_enc_input_categorical_features = enc.transform(list_input_categorical_features).toarray()		
	input_array = np.hstack((np.array(list_input_continuous_features),list_enc_input_categorical_features))
	return input_array, output_array
	
input_array, output_array = data_adult('/Users/tianchima/Desktop/adult.data')
print(input_array.shape)
