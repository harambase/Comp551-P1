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
def data_heart(data_path_heart,whether_One_hot_coding):
	with open(data_path_heart, 'r') as f:
		a = f.readlines()
	for i in range(len(a)):
		a[i] = a[i].split()
	input_array_original = np.empty([len(a),len(a[0])-1], dtype = float)
	output_array = np.empty([len(a)], dtype = int) 
	for i in range(len(a)):
		for j in range(len(a[0])-1):
			input_array_original[i][j] = float(a[i][j])
	for i in range(len(a)):
		if a[i][-1].strip() == '1':
			# 1 for absence of heart disease
			output_array[i] = 1  
			# 2 for presence of heart disease
		else:
			output_array[i] = 2
	list_input_continuous_features = []
	list_input_categorical_features = []
	for i in range(len(a)):
		list_input_continuous_features.append([a[i][0],a[i][3],a[i][4],a[i][7],a[i][9],a[i][11]])
		list_input_categorical_features.append([a[i][1],a[i][2],a[i][5],a[i][6],a[i][8],a[i][10],a[i][12]])
	enc = OneHotEncoder(drop = 'first')
	enc.fit(list_input_categorical_features)
	list_enc_input_categorical_features = enc.transform(list_input_categorical_features).toarray()		
	input_array = np.hstack((np.array(list_input_continuous_features),list_enc_input_categorical_features))
	if whether_One_hot_coding:
		return input_array, output_array
	else:
		return input_array_original, output_array

#'/Users/tianchima/Desktop/adult.data'
# test data = '/Users/tianchima/Desktop/adult.test'
def data_adult(data_path_adult, whether_One_hot_coding):
	with open(data_path_adult, 'r') as f:
		a = f.readlines()
	a = a[:-1]
	for i in range(len(a)):
		a[i] = a[i].split(',')
	for i in range(len(a)):
		for j in range(len(a[i])):
			a[i][j] = a[i][j].strip()
	list_workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
	list_education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
	list_maritalstatus = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
	list_occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
	list_relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
	list_race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
	list_sex = ['Female', 'Male']
	list_ativecountry = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
	long_list = [list_workclass, list_education, list_maritalstatus, list_occupation, list_relationship, list_race, list_sex, list_ativecountry]
	new_a = []
	for i in range(len(a)):
		if len(a[i])==15:
			if a[i][1] in list_workclass:
				if a[i][3] in list_education:
					if a[i][5] in list_maritalstatus:
						if a[i][6] in list_occupation:
							if a[i][7] in list_relationship:
								if a[i][8] in list_race:
									if a[i][9] in list_sex: 
										if a[i][13] in list_ativecountry:
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
		if new_a[i][-1] == '>50K' or new_a[i][-1] == '>50K.':
			output_array[i] = 1
		else:
			output_array[i] = 0
	enc = OneHotEncoder(categories = long_list, drop = 'first')
	enc.fit(list_input_categorical_features)
	list_enc_input_categorical_features = enc.transform(list_input_categorical_features).toarray()		
	input_array = np.hstack((np.array(list_input_continuous_features),list_enc_input_categorical_features))
	input_array_original = np.empty([len(new_a),len(new_a[0])-1], dtype = str)
	for i in range(len(new_a)):
		for j in range(len(new_a[0])-1):
			input_array_original[i][j] = str(new_a[i][j])
	if whether_One_hot_coding:
		return input_array, output_array
	else:
		return input_array_original, output_array

def split_into_train_test(input_array,output_array):
	output_array = output_array = output_array.reshape(input_array.shape[0],1)
	total_array = np.hstack((input_array,output_array))
	np.random.seed(2)
	np.random.shuffle(total_array)
	ratio_test = 0.15
	number_train_set = round(total_array.shape[0]*(1-ratio_test))
	#return train_input_arrray, train_output_array, test_input_array, test_output_array
	return total_array[:number_train_set,:][:,:-1], total_array[:number_train_set,:][:,-1],  total_array[number_train_set:,:][:,:-1],  total_array[number_train_set:,:][:,-1]

input_array, output_array = data_heart('/Users/tianchima/Desktop/heart.dat', True)
print(input_array.shape)
a,b,c,d = split_into_train_test(input_array, output_array)
print(c.shape)