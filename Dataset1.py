import numpy as np
data = 'ionosphere.data'
with open(data, 'r') as f:
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
		# 2 for bad
	else:
		output_array[i] = 0


print(input_array)