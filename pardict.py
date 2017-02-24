# General parameter interface information:
# Dictionary: par = parDict['C-N']
# Constructor: par = SkfInfo(name='C-N', paramPath='./mio-0-1')
# Getters:
#   par.GetSkInt(key='Hss0', dist=1.5)
#   par.GetAtomProp(key='Es')
#   par.GetRep(dist=1.5)

# SKF parameter interface:
import os
from skfinfo import SkfInfo

def ParDict(paramPath, elements=None):
    if elements is None:
        allFiles = os.listdir(paramPath)
        pairList = []
        for filename in allFiles:
            if filename.endswith('.skf'):
                pairList += [filename[:-4]]
        return {pair: SkfInfo(pair, paramPath) for pair in pairList}
    return {el1 + '-' + el2: SkfInfo(el1 + '-' + el2, paramPath)
            for el1 in elements for el2 in elements}
    
