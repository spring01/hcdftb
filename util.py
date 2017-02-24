
import numpy as np

def TriuToSymm(matrix):
    for ind in range(matrix.shape[0]):
        matrix[ind:, ind] = matrix[ind, ind:]
    return matrix

def BMatArr(matrix):
    return np.array(np.bmat(matrix))
