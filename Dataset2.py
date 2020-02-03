import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def createDataframe(file):
	# test data = '/Users/tianchima/Desktop/adult.test'
	with open(file, 'r') as f:
		a = f.readlines()
	a = a[:-1]
	for i in range(len(a)):
		a[i] = a[i].split(',')
	for i in range(len(a)):
		for j in range(len(a[i])):
			a[i][j] = a[i][j].strip()

	list_workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
					  'Without-pay',
					  'Never-worked']
	list_education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
					  '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
	list_maritalstatus = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
						  'Married-spouse-absent', 'Married-AF-spouse']
	list_occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
					   'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
					   'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
	list_relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
	list_race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
	list_sex = ['Female', 'Male']
	list_ativecountry = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
						 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
						 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
						 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
						 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
						 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

	new_a = []
	for i in range(len(a)):
		if len(a[i]) == 15:
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
	output_array = np.empty([len(new_a)], dtype=float)
	for i in range(len(new_a)):
		list_i1 = []
		list_i2 = []
		for j in range(len(new_a[i]) - 1):
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
	print(type(enc.transform(list_input_categorical_features)))
	list_enc_input_categorical_features = enc.transform(list_input_categorical_features).toarray()
	input_array = np.hstack((np.array(list_input_continuous_features), list_enc_input_categorical_features))
	input_array = pd.DataFrame(input_array).astype(float)
	input_array = (input_array - input_array.mean()) / (input_array.max() - input_array.min())
	output_array = pd.DataFrame({'104': output_array}).astype(float)
	df = pd.concat([input_array, output_array], axis=1)
	print(df)
	return df


class PrepareData:
	dataframe = createDataframe("adult.data")
	testdata = createDataframe("adult.test")
