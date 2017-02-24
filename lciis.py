import numpy as np

class LCIIS(object):
    
    def __init__(self, overlap, toOr, verbose=False, maxNumFock=60):
        self.__verbose = verbose
        from collections import deque
        self.__fList = deque(maxlen=maxNumFock)
        self.__trFList = deque(maxlen=maxNumFock)
        self.__trDList = deque(maxlen=maxNumFock)
        self.__toOr = toOr
        self.__orTo = overlap.dot(toOr)
        # self.__comm[i * numFock + j] = commutator [Fi, Dj]
        self.__comm = [None] * maxNumFock**2
        # self.__bigMat[i * numFock + j, k * numFock + l] = T(i, j, k, l)
        self.__bigMat = np.zeros((maxNumFock**2, maxNumFock**2))
        self.__maxIterNewton = 200
        self.__gradNormThres = 1e-12
    
    # Enqueue Fock list and density list, then extrapolate Fock
    # Note: this Fock list should be built from this density list
    #   fockList: [fock] for rhf/rks; [fockA, fockB] for uhf/uks
    #   densList: [dens] for rhf/rks; [densA, densB] for uhf/uks
    #   return: [newFock] for rhf/rks; [newFockA, newFockB] for uhf/uks
    def NewFock(self, fockList, densList):
        # enqueue fock & density and update commutators & tensor
        if len(self.__fList) == self.__fList.maxlen:
            self.__PreUpdateFull()
        else:
            self.__PreUpdateNotFull()
        self.__fList.append(fockList)
        shape = (self.__toOr.shape[0], -1)
        trFList = [self.__toOr.T.dot(fock.reshape(shape)) for fock in fockList]
        trDList = [dens.reshape(shape).dot(self.__orTo) for dens in densList]
        self.__trFList.append(trFList)
        self.__trDList.append(trDList)
        self.__UpdateCommBigMat()
        
        # coeff always has length numFock with 0.0 filled at the front in need
        numFock = len(self.__fList)
        shapeTensor = (numFock, numFock, numFock, numFock)
        tensor = self.__bigMat[:numFock**2, :numFock**2].reshape(shapeTensor)
        coeff = np.zeros(numFock)
        for numUse in range(len(tensor), 0, -1):
            tensorUse = tensor[-numUse:, -numUse:, -numUse:, -numUse:]
            (success, iniCoeffUse) = self.__InitialCoeffUse(tensorUse)
            if not success:
                print('__InitialCoeffUse failed; reducing tensor size')
                continue
            (success, coeffUse) = self.__NewtonSolver(tensorUse, iniCoeffUse)
            if not success:
                print('__NewtonSolver failed; reducing tensor size')
                continue
            else:
                coeff[-len(coeffUse):] = coeffUse
                break
        # end for
        if self.__verbose:
            print('  lciis coeff:')
            print('  ' + str(coeff).replace('\n', '\n  '))
        shape = self.__fList[0][0].shape
        return [coeff.dot([fList[spin].ravel()
                           for fList in self.__fList]).reshape(shape)
                for spin in range(len(self.__fList[0]))]
    
    def __PreUpdateFull(self):
        maxNumFock = self.__fList.maxlen
        for ind in range(1, maxNumFock):
            sourceFrom = ind * maxNumFock + 1
            sourceTo = (ind + 1) * maxNumFock
            shiftBy = -(maxNumFock + 1)
            source = slice(sourceFrom, sourceTo)
            target = slice(sourceFrom + shiftBy, sourceTo + shiftBy)
            self.__comm[target] = self.__comm[source]
            self.__bigMat[target, :] = self.__bigMat[source, :]
            self.__bigMat[:, target] = self.__bigMat[:, source]
    
    def __PreUpdateNotFull(self):
        numFock = len(self.__fList) + 1
        for ind in range(numFock - 1, 1, -1):
            sourceFrom = (ind - 1) * (numFock - 1)
            sourceTo = ind * (numFock - 1)
            shiftBy = ind - 1
            source = slice(sourceFrom, sourceTo)
            target = slice(sourceFrom + shiftBy, sourceTo + shiftBy)
            self.__comm[target] = self.__comm[source]
            self.__bigMat[target, :] = self.__bigMat[source, :]
            self.__bigMat[:, target] = self.__bigMat[:, source]
    
    def __UpdateCommBigMat(self):
        numFock = len(self.__fList)
        # update self.__comm
        update1 = range(numFock - 1, numFock**2 - 1, numFock)
        update2 = range(numFock**2 - numFock, numFock**2)
        for indComm in update1:
            self.__comm[indComm] = self.__Comm((indComm + 1) // numFock - 1, -1)
        for indComm in update2:
            self.__comm[indComm] = self.__Comm(-1, indComm - update2[0])
        # update self.__bigMat
        update = list(update1) + list(update2)
        full = slice(0, numFock**2)
        commFull = np.array(self.__comm[full])
        self.__bigMat[update, full] = commFull[update, :].dot(commFull.T)
        self.__bigMat[full, update] = self.__bigMat[update, full].T
    
    def __Comm(self, indFock, indDens):
        def Comm(trF, trD):
            trFdotOrD = trF.dot(trD)
            comm = trFdotOrD - trFdotOrD.T
            return comm[np.triu_indices_from(comm, 1)]
        zipList = zip(self.__trFList[indFock], self.__trDList[indDens])
        return np.concatenate([Comm(trF, trD) for (trF, trD) in zipList])
    
    # return (success, cdiis_coefficients)
    def __InitialCoeffUse(self, tensorUse):
        numUse = len(tensorUse)
        ones = np.ones((numUse, 1))
        hess = np.zeros((numUse, numUse))
        # hess[i, i] = tensorUse[i, i, j, j]
        for ind in range(numUse):
            hess[ind, :] = np.diag(tensorUse[ind, ind, :, :])
        hessLag = np.bmat([[hess,   ones   ],
                           [ones.T, [[0.0]]]])
        gradLag = np.concatenate((np.zeros(numUse), [1.0]))
        iniCoeffUse = np.linalg.solve(hessLag, gradLag)[0:-1]
        return (not np.isnan(sum(iniCoeffUse)), iniCoeffUse)
    
    # return (success, lciis_coefficients)
    def __NewtonSolver(self, tensorUse, coeffUse):
        tensorGrad = tensorUse + tensorUse.transpose(1, 0, 2, 3)
        tensorHess = tensorGrad + tensorUse.transpose(0, 2, 1, 3)
        tensorHess += tensorUse.transpose(3, 0, 1, 2)
        tensorHess += tensorUse.transpose(0, 3, 1, 2)
        tensorHess += tensorUse.transpose(1, 3, 0, 2)
        ones = np.ones((len(coeffUse), 1))
        value = np.inf
        for _ in range(self.__maxIterNewton):
            (grad, hess) = self.__GradHess(tensorGrad, tensorHess, coeffUse)
            oldValue = value
            value = self.__Value(tensorUse, coeffUse)
            gradLag = np.concatenate((grad, [0.0]))
            hessLag = np.bmat([[hess,      ones],
                               [ones.T, [[0.0]]]])
            step = np.linalg.solve(hessLag, gradLag)
            if np.isnan(sum(step)):
                print('Inversion failed')
                return (False, coeffUse)
            coeffUse -= step[0:-1]
            if np.abs(value - oldValue) < self.__gradNormThres:
                return (True, coeffUse)
        # end for
        print('Newton did not converge')
        return (False, coeffUse)
    
    def __GradHess(self, tensorGrad, tensorHess, coeffUse):
        numUse = len(coeffUse)
        grad = tensorGrad.reshape(-1, numUse).dot(coeffUse)
        grad = grad.reshape(-1, numUse).dot(coeffUse)
        grad = grad.reshape(-1, numUse).dot(coeffUse)
        hess = tensorHess.reshape(-1, numUse).dot(coeffUse)
        hess = hess.reshape(-1, numUse).dot(coeffUse)
        hess = hess.reshape(-1, numUse)
        return (grad, hess)
    
    def __Value(self, tensorUse, coeffUse):
        numUse = len(coeffUse)
        value = tensorUse.reshape(-1, numUse).dot(coeffUse)
        value = value.reshape(-1, numUse).dot(coeffUse)
        value = value.reshape(-1, numUse).dot(coeffUse)
        value = value.dot(coeffUse)
        return value

