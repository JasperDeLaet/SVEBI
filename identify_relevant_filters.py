import numpy as np
import spams
import pickle
import os

'''
This file implements the optimization procedure of SVEBI. The spams library implements the
the mu-lasso optimization function
'''

output_dir = './output/w/'


if not os.path.exists(output_dir):
    os.mkdir(output_dir)

number_of_classes = 50

activations = None
labels = None
'''
We load X and L for each class and concatenate them.
'''
for class_index in range(number_of_classes):
    with open('./output/activation_maps/' + str(class_index) + '.npy', 'rb') as file_add:
        class_info = pickle.load(file_add)
    if activations is None:
        activations = class_info[0]
        labels = np.asarray(class_info[1])
    else:
        activations = np.concatenate((activations, np.asarray(class_info[0])), axis=0)
        labels = np.concatenate((labels, np.asarray(class_info[1])), axis=0)

indices = np.arange(activations.shape[0])
np.random.shuffle(indices)

activations = activations[indices, :]
labels = labels[indices, :]

'''
Here it is necessary to change the type to fortran as it is needed by the library that implements
the lasso optimization function.
'''
X = np.asfortranarray(activations)
L = np.asfortranarray(labels)

del activations
del labels

# X refers to the L matrix
# D refers to the X matrix
# 'lambda1' refers to mu from the mu-lasso optimization
w = spams.lasso(X=L, D=X, lambda1=50, mode=0, pos=True, numThreads=5)
w = np.swapaxes(w.toarray(), 0, 1)

# Save w
with open(output_dir + 'w.npy', 'wb') as file_add:
    pickle.dump(w, file_add)
del w
del X
del L
