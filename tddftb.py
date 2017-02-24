
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigsh

EPS = 1e-10

class TDDFTB(object):
    
    def __init__(self, dftb):
        goundEnergy, fockList, densList = dftb.SCF()
        if len(fockList) == 1 and len(densList) == 1:
            fockList *= 2
            densList *= 2
        scfSolution = [dftb.SolveFock(fock) for fock in fockList]
        orbEnergyList = [sol[0] for sol in scfSolution]
        orbList = [sol[1] for sol in scfSolution]
        overlap = dftb.GetOverlap()
        coreH = dftb.GetCoreH()
        gamma = dftb.GetGamma()
        doublju = dftb.GetDoublju()
        
        numBasis = overlap.shape[0]
        numElecAB = dftb.GetNumElecAB()
        orbTildeList = [overlap.dot(orb) for orb in orbList]
        
        dimList = [ne * (numBasis - ne) for ne in numElecAB]
        bigPTildeList = [lil_matrix((numBasis, dim)) for dim in dimList]
        zipList = zip(bigPTildeList, orbList, orbTildeList, numElecAB)
        for bigPT, orb, orbT, ne in zipList:
            for i in range(ne):
                for a in range(ne, numBasis):
                    bigPTIndex = i * (numBasis - ne) + a - ne
                    values = orb[:, i] * orbT[:, a] + orb[:, a] * orbT[:, i]
                    values *= 0.5
                    values[np.abs(values) < EPS] = 0.0
                    bigPT[:, bigPTIndex] = values[:, np.newaxis]
        #~ print bigPTildeList
        shellBasis = dftb.GetShellBasis()
        numShell = len(shellBasis)
        smallQList = [lil_matrix((numShell, dim)) for dim in dimList]
        for smallQ, bigPT in zip(smallQList, bigPTildeList):
            for iShell, bas in enumerate(shellBasis):
                values = bigPT[bas, :].sum(axis=0)
                values[np.abs(values) < EPS] = 0.0
                smallQ[iShell, :] = values
        #~ print smallQ
        gpd = lil_matrix(gamma + doublju)
        gmd = lil_matrix(gamma - doublju)
        bigKAA = _DiscardZeros(smallQList[0].T.dot(gpd).dot(smallQList[0]))
        bigKAB = _DiscardZeros(smallQList[0].T.dot(gmd).dot(smallQList[1]))
        bigKBA = _DiscardZeros(smallQList[1].T.dot(gmd).dot(smallQList[0]))
        bigKBB = _DiscardZeros(smallQList[1].T.dot(gpd).dot(smallQList[1]))
        #~ print bigKAA.shape
        bigK = (bmat([[bigKAA, bigKAB], [bigKBA, bigKBB]]))
        #~ print bigK
        
        vecO = sum([[ea - ei for ei in orbE[:ne] for ea in orbE[ne:]]
                    for orbE, ne in zip(orbEnergyList, numElecAB)], [])
        #~ print vecO
        #~ print diags(vecO, offsets=0)
        bigAPB = diags(vecO, offsets=0) + 2.0 * bigK
        #~ print bigAPB
        
        scaling = np.sqrt(vecO)
        
        cscBigAPB = bigAPB.tocsc()
        cscBigAPB.data *= np.take(scaling, cscBigAPB.indices)
        
        csrBigAPB = cscBigAPB.T
        csrBigAPB.data *= np.take(scaling, csrBigAPB.indices)
        
        #~ numEigVals = csrBigAPB.shape[0]
        #~ bigOmega = lil_matrix((numEigVals + 2, numEigVals + 2))
        #~ keep = np.abs(csrBigAPB) > EPS
        #~ bigOmega[keep] = csrBigAPB[keep]
        numEigVals = 20
        bigOmega = csrBigAPB
        #~ print dimList
        #~ print eigsh(bigOmega, k=numEigVals)[0]
        print np.sqrt(eigsh(bigOmega, k=numEigVals, sigma=0.0, return_eigenvectors=False)) * 27.2114
        
        #~ print bigAPB.todense()
        #~ print 'step 1 ref'
        #~ print np.array(bigAPB.todense()) * scaling
        #~ print 'step 1 val'
        #~ print cscBigAPB.T.todense()
        #~ print type(cscBigAPB.T)
        
        #~ bigOmega = (bigAPB * scaling).T * scaling
        #~ print type(bigOmega)
        #~ print bigOmega
        #~ print np.linalg.eigh(bigOmega)[0]
        #~ print np.sqrt(np.linalg.eigh(bigOmega)[0]) * 27.2114
    
def _DiscardZeros(spMat):
    newSpMat = lil_matrix(spMat.shape)
    keep = np.abs(spMat) > EPS
    newSpMat[keep] = spMat[keep]
    return newSpMat

