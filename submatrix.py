import numpy as np
from scipy.io import loadmat

data = loadmat("values2.mat")
submatrix_folder = data["values2"]

given_matrix = np.loadtxt("submatrix_1.csv", delimiter=",")

for i in range(submatrix_folder.shape[0]):
    if np.array_equal(submatrix_folder[i], given_matrix):
        print("submatrix index",i)
        print("Value:",submatrix_folder[i])
        break