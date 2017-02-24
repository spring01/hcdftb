import numpy as np

class GeomOpt(object):
    
    #--------------------------------------------------------------------------
    # Either info['GradFunc'] or info['EnergyFunc'] has to be provided;
    # info['GradFunc'] is preferred over info['EnergyFunc']
    # with info['EnergyFunc'] the (expensive) finite difference is invoked
    #
    # Function interfaces:
    #   'EnergyFunc':   (energy, densList) = EnergyFunc(xyz, guess, info)
    #
    #   'GradFunc'  :   (energy, densList, grad) = GradFunc(xyz, guess, info)
    #
    # Input:
    #   xyz:        2d np.array with shape (numAtom, 4);
    #               1st column is atomic number, other 3's are xyz coordinates
    #
    #   guess:      'core', 'sad', or list of density matrices (densList);
    #               see Output: densList for details
    #
    #   info:       Dictionary with at least 'GradFunc' or 'EnergyFunc';
    #               can be used to pass other arguments to GradFunc/EnergyFunc
    #
    #   verbose:    True or False for printing
    #
    # Output:
    #   energy:     Double precision scalar; total scf energy
    #
    #   densList:   List of 1 or 2 density matrices (wavefunction);
    #               [dens] for RHF/RKS, [densA, densB] for UHF/UKS
    #
    #   grad:       2d np.array with shape (numAtom, 3)
    #--------------------------------------------------------------------------
    def __init__(self, xyz, info, verbose=False):
        self.__xyz = np.array(xyz)
        self.__info = info
        self.__verbose = verbose
        self.__energyFunc = info['EnergyFunc']
        if 'GradFunc' in info:
            self.__gradFunc = info['GradFunc']
        else:
            self.__gradFunc = self.__FiniteDiff
        self.__maxNumIter = 200
        self.__iniStepSize = 1.0
        self.__stepSizeShrinkBy = 0.8
        self.__thresMaxGrad = 0.000450
        self.__thresRmsGrad = 0.000300
        self.__thresMaxDisp = 0.001800
        self.__thresRmsDisp = 0.001200
    
    # Homemade BFGS optimizer
    def RunGeomOpt(self, guess='core'):
        xyz = self.__xyz
        info = self.__info
        (energy, densList, grad) = self.__gradFunc(xyz, guess, info)
        grad = grad.ravel()
        invHess = np.eye(grad.size)
        for numIter in range(1, self.__maxNumIter):
            if self.__verbose:
                print('geom opt iter {}; energy: {}'.format(numIter, energy))
            direction = -invHess.dot(grad)
            stepSize = self.__LineSearch(direction, grad, xyz, energy, densList)
            step = stepSize * direction
            xyz[:, 1:] += step.reshape(xyz[:, 1:].shape)
            if self.__Converged(grad, step):
                break
            gradOld = grad
            (energy, densList, grad) = self.__gradFunc(xyz, densList, info)
            grad = grad.ravel()
            diff = grad - gradOld
            stepDiff = step.dot(diff)
            infHessDiff = invHess.dot(diff)
            coeff1 = stepDiff + diff.dot(infHessDiff)
            invHessDiffOutStep = np.outer(infHessDiff, step)
            invHess += coeff1 * np.outer(step, step) / stepDiff**2
            invHess -= (invHessDiffOutStep + invHessDiffOutStep.T) / stepDiff
        # end for
        if self.__verbose:
            print('geom opt done; energy: {}'.format(energy))
        return (xyz, energy)
    
    def __LineSearch(self, direction, grad, xyz, energy, densList):
        stepSize = self.__iniStepSize
        shrinkBy = self.__stepSizeShrinkBy
        densListT = [dens.copy() for dens in densList]
        while True:
            step = stepSize * direction
            xyzT = xyz.copy()
            xyzT[:, 1:] += step.reshape(xyz[:, 1:].shape)
            energyT, densListT = self.__energyFunc(xyzT, densListT, self.__info)
            if energyT <= energy + step.dot(grad):
                break
            stepSize *= shrinkBy
        return stepSize
    
    # SciPy's optimizer wrapper
    def RunGeomOptSciPy(self, guess='core'):
        def Fun(coords, *args):
            xyz = args[0].copy()
            info = args[1]
            xyz[:, 1:] = coords.reshape(xyz[:, 1:].shape)
            guess = info['guess']
            (energy, info['guess'], grad) = self.__gradFunc(xyz, guess, info)
            return (energy, grad.ravel())
        class NumIter:
            numIter = 0
        def Callback(coords):
            NumIter.numIter += 1
            step = np.abs(coords - xyz[:, 1:].ravel())
            xyz[:, 1:] = coords.reshape(xyz[:, 1:].shape)
            maxDisp = np.max(np.abs(step))
            rmsDisp = np.sqrt(np.mean(step**2))
            if self.__verbose:
                form = '  {:s}: {:0.6f}'.format
                print('geom opt iter {}'.format(NumIter.numIter))
                print(form('maxDisp', maxDisp))
                print(form('rmsDisp', rmsDisp))
        # end Callback
        import scipy.optimize as opt
        xyz = self.__xyz.copy()
        info = self.__info.copy()
        info['guess'] = guess
        coordsIni = xyz[:, 1:].ravel().copy()
        optResult = opt.minimize(fun=Fun, x0=coordsIni, args=(xyz, info),
                                 method='BFGS', jac=True, callback=Callback)
        xyz[:, 1:] = optResult.x.reshape(xyz[:, 1:].shape)
        return (xyz, optResult.fun)
    
    # Finite (forward) difference approximation of gradient
    def __FiniteDiff(self, xyz, guess, info):
        (energy, densList) = self.__energyFunc(xyz, guess, info)
        diff = 1.0e-6
        grad = np.zeros((xyz.shape[0], xyz.shape[1] - 1))
        for atom in range(xyz.shape[0]):
            for coord in range(1, xyz.shape[1]):
                xyzNew = xyz.copy()
                xyzNew[atom, coord] += diff
                energyNew = self.__energyFunc(xyzNew, densList, info)[0]
                grad[atom, coord - 1] = (energyNew - energy) / diff
            # end For
        # end For
        return (energy, densList, grad)
    
    # Test geometry optimization convergence
    def __Converged(self, gradNew, step):
        maxGrad = np.max(np.abs(gradNew))
        rmsGrad = np.sqrt(np.mean(gradNew**2))
        maxDisp = np.max(np.abs(step))
        rmsDisp = np.sqrt(np.mean(step**2))
        if self.__verbose:
            form = '  {:s}: {:0.6f}'.format
            print(form('maxGrad', maxGrad) + form('thres', self.__thresMaxGrad))
            print(form('rmsGrad', rmsGrad) + form('thres', self.__thresRmsGrad))
            print(form('maxDisp', maxDisp) + form('thres', self.__thresMaxDisp))
            print(form('rmsDisp', rmsDisp) + form('thres', self.__thresRmsDisp))
        return (maxGrad < self.__thresMaxGrad and
                rmsGrad < self.__thresRmsGrad and
                maxDisp < self.__thresMaxDisp and
                rmsDisp < self.__thresRmsDisp)
    
