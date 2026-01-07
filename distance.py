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

target = 1.5 * sub
def manhattan_distance(target, main, skip_index=None):
    distances = []
    for i in range(main.shape[0]):
        if i == skip_index:
            distances.append(np.inf)
        else:
            distances.append(np.sum(np.abs(main[i] - target)))
    return np.array(distances)

d1 = manhattan_distance(target, main, sub_matrix_index)
idx1 = np.argmin(d1)

print("Manhattan Closest Index:", idx1)
print("Distance:", d1[idx1])

def cosine_distance(target, main, skip_index=None):
    t = target.flatten()
    distances = []
    for i in range(main.shape[0]):
        if i == skip_index:
            distances.append(np.inf)
        else:
            m = main[i].flatten()
            cos_sim = np.dot(t, m) / (np.linalg.norm(t) * np.linalg.norm(m))
            distances.append(1 - cos_sim)   # cosine distance
    return np.array(distances)

d3 = cosine_distance(target, main, sub_matrix_index)
idx3 = np.argmin(d3)

print("Cosine Closest Index:", idx3)
print("Distance:", d3[idx3])
def minkowski_distance(target, main, p=3, skip_index=None):
    distances = []
    for i in range(main.shape[0]):
        if i == skip_index:
            distances.append(np.inf)
        else:
            distances.append(np.power(np.sum(np.abs(main[i] - target) ** p), 1/p))
    return np.array(distances)

d4 = minkowski_distance(target, main, p=3, skip_index=sub_matrix_index)
idx4 = np.argmin(d4)

print("Minkowski Closest Index:", idx4)
print("Distance:", d4[idx4])
