import numpy as np
from collections import deque

class CDIIS(object):
    
    def __init__(self, overlap, maxNumFock=20):
        self.__overlap = overlap
        self.__fockQue = deque(maxlen=maxNumFock)
        self.__commQue = deque(maxlen=maxNumFock)
    
    def NewFock(self, fockList, comm):
        self.__fockQue.append(fockList)
        self.__commQue.append(comm)
        commMat = np.array(self.__commQue)
        ones = np.ones((len(self.__commQue), 1))
        cdiisMat = np.bmat([[commMat.dot(commMat.T),   ones            ],
                            [ones.T                ,   np.zeros((1, 1))]])
        rightSide = np.array([0.0] * len(self.__commQue) + [1.0])
        coeff = np.linalg.solve(cdiisMat, rightSide)[0:-1]
        sh = self.__fockQue[0][0].shape
        return [coeff.dot([fl[sp].ravel() for fl in self.__fockQue]).reshape(sh)
                for sp in range(len(fockList))]
    

