import numpy as np
from scipy.io import loadmat

data = loadmat("values2.mat")
main = data["values2"]     


sub = np.loadtxt("submatrix_1.csv", delimiter=",")


sub_matrix_index = None

for i in range(len(main)):
    if np.allclose(main[i], sub):
        sub_matrix_index = i
        break

higher_50_percent = sub + (0.50 * sub)
def euclidean_distances(target, main, skip_index=None):
    distances = []

    for i in range(main.shape[0]):
        if i == skip_index:
            distances.append(np.inf)  
        else:
            distances.append(np.linalg.norm(main[i] - target))

    return np.array(distances)

distances_50_percent = euclidean_distances(
    higher_50_percent,
    main,
    skip_index=sub_matrix_index
)

closest_index_50 = np.argmin(distances_50_percent)

print("Submatrix index:", sub_matrix_index)
print("50% HIGHER DEVIANT closest index:", closest_index_50)
print("Euclidean distance:", distances_50_percent[closest_index_50])
print("Closest submatrix:\n", main[closest_index_50])