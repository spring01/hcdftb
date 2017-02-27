
import numpy as np
from atomentry import AtomEntry
from repulsion import Repulsion
from matelem import Overlap, CoreH
from sccparam import Gamma, Doublju
from cdiis import CDIIS

ANGSTROM2BOHR = 1.889725989

class DFTB(object):
    
    __maxSCFIter = 30
    __thresRms = 1.0e-8
    __thresMax = 1.0e-6
    
    def __init__(self, parDict, cart, rep='def', charge=0, mult=1,
                 lenUnit='Angstrom', verbose=False):
        self.__verbose = verbose
        # Any internal length is in Bohr
        cartBohr = np.array(cart)
        cartBohr[:, 1:] *= {'Bohr': 1.0, 'Angstrom': ANGSTROM2BOHR}[lenUnit]
        self.atomList = [AtomEntry(cartRow) for cartRow in cartBohr]
        self.parDict = parDict
        argTup = self.atomList, parDict
        self.repObj = Repulsion(*argTup) if rep == 'def' else rep(cartBohr)
        
        #~ import pdb; pdb.set_trace()
        
        self.repulsion = self.repObj.Energy()
        self.__shellBasis = _ShellBasis(self.atomList)
        self.__numVEShellNeu = _NumVEShellNeu(*argTup)
        self.__numElecAB = _NumElecAB(sum(self.__numVEShellNeu) - charge, mult)
        self.__overlap = Overlap(*argTup).Matrix()
        self.__coreH = CoreH(*argTup).Matrix()
        self.__gamma = Gamma(*argTup).Matrix()
        self.__doublju = Doublju(*argTup).Matrix() if mult > 1 else None
        self.__toOrtho = self.__ToOrtho(self.__overlap)
        self.solFockList = None
    
    @staticmethod
    def Energy(cart, guess, info):
        parDict = info['parDict']
        charge = info['charge']
        mult = info['mult']
        rep = info['rep'] if 'rep' in info else 'def'
        verbose = info['verbose'] if 'verbose' in info else False
        dftb = DFTB(parDict, cart, rep=rep,
                    charge=charge, mult=mult, verbose=verbose)
        if 'lineSearchThresRms' in info:
            dftb.__thresRms = info['lineSearchThresRms']
        if 'lineSearchThresMax' in info:
            dftb.__thresMax = info['lineSearchThresMax']
        energy, _, densList = dftb.SCF(guess)
        energy += dftb.repulsion
        return energy, densList
    
    @staticmethod
    def EnergyGrad(cart, guess, info):
        parDict = info['parDict']
        charge = info['charge']
        mult = info['mult']
        rep = info['rep'] if 'rep' in info else 'def'
        verbose = info['verbose'] if 'verbose' in info else False
        dftb = DFTB(parDict, cart, rep=rep,
                    charge=charge, mult=mult, verbose=verbose)
        energy, _, densList = dftb.SCF(guess)
        energy += dftb.repulsion
        grad = dftb.ElecEnergyXYZDeriv1()
        grad += dftb.RepulsionXYZDeriv1()
        print cart
        return energy, densList, grad
    
    def SCF(self, guess='core'):
        if type(guess) is str:
            guessDict = {'core': self.__GuessCore}
            densList = guessDict[guess.lower()]()
        elif type(guess) is list:
            densList = guess
        cdiis = CDIIS(self.__overlap)
        for numIter in range(1, self.__maxSCFIter + 1):
            oldDensList = densList
            fockList = self.__DensToFock(densList)
            comm = self.__Comm(fockList, densList)
            fockList = cdiis.NewFock(fockList, comm)
            densList = self.__FockToDens(fockList)
            if self.__verbose:
                print 'scf iter %d' % numIter
                print self.__coreH.ravel().dot(sum(densList).ravel()) * 2
            if self.__Converged(densList, oldDensList, comm):
                break
        if self.__verbose:
            print 'SCF done in %d iterations.' % numIter
        elecEnergy = self.__ElecEnergy(densList)
        if numIter >= self.__maxSCFIter:
            print 'Not converged.'
            elecEnergy = np.nan
        self.solFockList = fockList
        return elecEnergy, fockList, densList
    
    def ElecEnergyXYZDeriv1(self):
        argTup = self.atomList, self.parDict
        overlapXYZDeriv1 = Overlap(*argTup).Deriv1List()
        coreHXYZDeriv1 = CoreH(*argTup).Deriv1List()
        gammaXYZDeriv1 = Gamma(*argTup).Deriv1List()
        if self.solFockList is None:
            self.SCF()
        solList = [self.SolveFock(fock) for fock in self.solFockList]
        occSolList = [(ev[:ne], orb[:, :ne])
                      for (ev, orb), ne in zip(solList, self.__numElecAB)]
        factor = 2.0 / len(self.solFockList)
        densList = [occOrb.dot(occOrb.T) for _, occOrb in occSolList]
        dens = sum(densList) * factor
        densDiff = densList[0] - densList[-1]
        enDens = sum([(orb * ev).dot(orb.T) for ev, orb in occSolList]) * factor
        deltaQShell, magQShell = self.__DeltaQShellMagQShell(densList)
        dftb2eCou = self.__Dftb2eMatrix(self.__gamma, deltaQShell)
        if self.__doublju is not None:
            dftb2eExc = self.__Dftb2eMatrix(self.__doublju, magQShell)
        numAtom = len(self.atomList)
        deriv1 = np.zeros((numAtom, 3))
        for ind in range(numAtom):
            basis = coreHXYZDeriv1[ind]['basisInd']
            shell = gammaXYZDeriv1[ind]['shellInd']
            densPart = dens[basis, :].ravel()
            enDensPart = enDens[basis, :].ravel()
            densDiffPart = densDiff[basis, :].ravel()
            dftb2eCouPart = dftb2eCou[basis, :].ravel()
            if self.__doublju is not None:
                dftb2eExcPart = dftb2eExc[basis, :].ravel()
            for ci in range(3):
                coreHDeriv1 = coreHXYZDeriv1[ind]['xyz'][ci].ravel()
                partH = densPart.dot(coreHDeriv1)
                gammaDeriv1 = gammaXYZDeriv1[ind]['xyz'][ci]
                partGamma = gammaDeriv1.dot(deltaQShell).dot(deltaQShell[shell])
                overlapDeriv1 = overlapXYZDeriv1[ind]['xyz'][ci].ravel()
                couOver = dftb2eCouPart * overlapDeriv1
                part2e = densPart.dot(couOver)
                if self.__doublju is not None:
                    excOver = dftb2eExcPart * overlapDeriv1
                    part2e += densDiffPart.dot(excOver)
                partS = enDensPart.dot(overlapDeriv1)
                deriv1[ind, ci] = 2.0 * (partH + part2e - partS) + partGamma
        return deriv1
    
    def RepulsionXYZDeriv1(self):
        return self.repObj.XYZDeriv1()
    
    def SolveFock(self, fock):
        orFock = self.__toOrtho.T.dot(fock).dot(self.__toOrtho)
        orbEigVal, orOrb = np.linalg.eigh(orFock)
        argsort = np.argsort(orbEigVal)
        return (orbEigVal[argsort], self.__toOrtho.dot(orOrb[:, argsort]))
    
    def GetOverlap(self):
        return self.__overlap
    
    def GetCoreH(self):
        return self.__coreH
    
    def GetGamma(self):
        return self.__gamma
    
    def GetDoublju(self):
        return self.__doublju
    
    def GetNumElecAB(self):
        return self.__numElecAB
    
    def GetShellBasis(self):
        return self.__shellBasis
    
    # Find a transform from atomic orbitals to orthogonal orbitals
    def __ToOrtho(self, overlap):
        (eigVal, eigVec) = np.linalg.eigh(overlap)
        keep = eigVal > 1.0e-6
        return eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]
    
    def __GuessCore(self):
        return self.__FockToDens([self.__coreH] * len(set(self.__numElecAB)))
    
    def __FockToDens(self, fockList):
        orbList = [self.SolveFock(fock)[1] for fock in fockList]
        occOrbList = [orb[:, :ne] for orb, ne in zip(orbList, self.__numElecAB)]
        return [occOrb.dot(occOrb.T) for occOrb in occOrbList]
    
    def __Comm(self, fockList, densList):
        fdsList = [fock.dot(dens).dot(self.__overlap)
                   for fock, dens in zip(fockList, densList)]
        indices = np.triu_indices_from(self.__overlap, 1)
        return np.concatenate([(fds - fds.T)[indices] for fds in fdsList])
    
    def __DensToFock(self, densList):
        deltaQShell, magQShell = self.__DeltaQShellMagQShell(densList)
        couMat = self.__overlap * self.__Dftb2eMatrix(self.__gamma, deltaQShell)
        fockNoSpin = self.__coreH + couMat
        if self.__doublju is None:
            return [fockNoSpin]
        excMat = self.__overlap * self.__Dftb2eMatrix(self.__doublju, magQShell)
        return [fockNoSpin + exc for exc in [excMat, -excMat][:len(densList)]]
    
    def __ElecEnergy(self, densList):
        deltaQShell, magQShell = self.__DeltaQShellMagQShell(densList)
        factor = 2.0 / len(densList)
        energy = self.__coreH.ravel().dot(sum(densList).ravel()) * factor
        energy2e = self.__Dftb2eEnergy(self.__gamma, deltaQShell)
        if self.__doublju is not None:
            energy2e += self.__Dftb2eEnergy(self.__doublju, magQShell)
        return energy + energy2e
    
    def __DeltaQShellMagQShell(self, densList):
        # Mulliken population
        mlkPopList = [dens * self.__overlap for dens in densList]
        qShellSpin = np.array([[np.sum(pop[bas, :]) for pop in mlkPopList]
                               for bas in self.__shellBasis])
        # Total and delta charge
        qShell = np.sum(qShellSpin, axis=1) * 2.0 / len(densList)
        deltaQShell = qShell - self.__numVEShellNeu
        # Magnetization charge
        magQShell = (qShellSpin[:, 0] - qShellSpin[:, -1]).ravel()
        return deltaQShell, magQShell
    
    def __Dftb2eMatrix(self, shellMat, shellVec):
        zipList = zip(0.5 * shellMat.dot(shellVec), self.__shellBasis)
        epBasis = np.array([sum([[ep] * len(bas) for ep, bas in zipList], [])])
        return epBasis + epBasis.T
    
    def __Dftb2eEnergy(self, shellMat, shellVec):
        return 0.5 * shellMat.ravel().dot(np.outer(shellVec, shellVec).ravel())
    
    def __Converged(self, densList, oldDensList, comm=None):
        useComm = comm is not None
        zipList = zip(densList, oldDensList)
        diffDens = np.concatenate([dens - old for (dens, old) in zipList])
        rmsDens = np.sqrt(np.mean(diffDens**2))
        maxDens = np.max(np.abs(diffDens))
        rmsComm = np.sqrt(np.mean(comm**2)) if useComm else 0.0
        maxComm = np.max(np.abs(comm)) if useComm else 0.0
        if self.__verbose:
            form = '  {:s}: {:.3e}'.format
            print (form('rmsDiffDens', rmsDens) +
                   form('thres', self.__thresRms))
            print (form('maxDiffDens', maxDens) +
                   form('thres', self.__thresMax))
            if useComm:
                print (form('rmsComm', rmsComm) +
                       form('thres', self.__thresRms))
                print (form('maxComm', maxComm) +
                       form('thres', self.__thresMax))
        densThres = rmsDens < self.__thresRms and maxDens < self.__thresMax
        commThres = rmsComm < self.__thresRms and maxComm < self.__thresMax
        return densThres and commThres

def _ShellBasis(atomList):
    shellBasis = []
    for atom in atomList:
        for nShell in range(atom.numShell):
            curInd = sum([len(bas) for bas in shellBasis])
            shellBasis += [range(curInd, curInd + 2 * nShell + 1)]
    return shellBasis

def _NumVEShellNeu(atomList, parDict):
    numVEShellNeu = []
    for atom in atomList:
        par = parDict[atom.elem + '-' + atom.elem]
        sShell = [par.GetAtomProp('fs')]
        pShell = [par.GetAtomProp('fp')] if 'p' in atom.orbType else []
        dShell = [par.GetAtomProp('fd')] if 'd' in atom.orbType else []
        numVEShellNeu += sShell + pShell + dShell
    return numVEShellNeu

def _NumElecAB(numElecTotal, mult):
    numElecA = (numElecTotal + mult - 1) / 2.0
    if numElecA % 1 != 0.0:
        raise Exception('numElecTotal %d and mult %d ??' % (numElecTotal, mult))
    return [int(numElecA), int(numElecTotal - numElecA)]

