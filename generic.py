#--Decorators for functions
class FuncDecorator():

    def timeOperation(original_function):
        """
        Print elapsed time of 1 time operation.
        """
        def new_function(*args,**kwargs):
            import datetime
   
            before = datetime.datetime.now()                     
            original_output = original_function(*args,**kwargs)                
            after = datetime.datetime.now()           
            print('Elapsed Time = {0}'.format(after - before))

            return original_output
        return new_function


#--Decorators for classes
class ClsDecorator():

    def prohibitAttrSetter(cls):
        """
        Prohibit access to attribute setter
        """
        def setattr(self, key, value):
            class ProhibittedOperation(Exception): pass
            raise ProhibittedOperation('Not allowed to modify attributes directly.')

        cls.__setattr__ =  setattr
        return cls


#--Universal container
class UniversalContainer():
    """
    Usage
    - Print object to see all key and data (recursive).
    - listKey() shows all attribute keys of this object (current level).
    - listMethod() shows all methods of this object.
    """

    def __repr__(self, level=0):
        keys = [item for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]
        rep = []

        for key in keys:
            attr = getattr(self, key)
            if isinstance(attr, UniversalContainer):
                rep.append('-' * 3 * level + '.' + key)
                rep.append(attr.__repr__(level + 1))
            else:
                rep.append('-' * 3 * level + '.' + key)
                rep.append('-' * 3 * level + ' ' + str(attr))

        return '\n'.join(rep)

    def listKey(self):
        return [item for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]

    def listMethod(self):
        return [item for item in dir(self) if callable(getattr(self, item)) and not item.startswith("__")]


#--Setting container
@ClsDecorator.prohibitAttrSetter
class SettingContainer(UniversalContainer):
    """
    Usage
    - Convenient keyword = parameter setting.
    - Protected attribute setter. Use `update(key=value)` to modify content.
    """

    def update(self, **kwarg):
        for key, value in kwarg.items():
            self.__dict__[key] = value

    __init__ = update


#--Convert data to object form (recursive)
class ConvertedContainer(UniversalContainer):
    """
    Usage
    - Convert dict to object form (recursive).
    """
    
    def __new__(cls, data):
        from collections import Iterable

        if isinstance(data, dict):
            return super().__new__(cls)
        elif isinstance(data, Iterable) and not isinstance(data, str):
            return type(data)(ConvertedContainer(value) for value in data)
        else:
            return data

    def __init__(self, data):
        for i in range(len(data.keys())):
            setattr(self, list(data.keys())[i], ConvertedContainer(list(data.values())[i]))


#--Import config files as an object
def getConfigObj(path):
    """
    Read config files into an obj container.
    - Support file type: .json, .yml.
    """
    import yaml
    import re

    class UnknownFileType(Exception): pass

    with open(path, 'r', encoding='utf-8') as f:
        ext = re.search('\.(.+)', path).group(1)
        if ext == 'json': config_dic = json.load(f)
        elif ext == 'yml': config_dic = yaml.load(f)
        else: raise UnknownFileType('\'{}\' is not a supported file type.'.format(ext))

    return ConvertedContainer(config_dic)


#--Flatten list (recursive)
#Parameter: l, a list
#Return: a flattened list as a generator
def flattenList(l):
    import collections

    for el in l:
        if isinstance(el, collections.Sequence) and not isinstance(el, (str, bytes)):
            yield from flattenList(el)
        else:
            yield el


#--Element-wise list operation
#Return: operated list
def listEWiseOp(op, l):
    return list(map(op, l))


#--Graphing 2-D scatter plot
#With distribution and linear fitting line
def scatter(vectors, names):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    sns.set(color_codes=True)

    df = pd.DataFrame({
        names[0]: vectors[0],
        names[1]: vectors[1]
    })

    g = sns.jointplot(x=names[0], y=names[1], data=df, color="m", kind="reg", scatter_kws={"s": 10})
    plt.show()


#--Evaluate model with mse, cor, and graphing
def evalModel(predictions, truth, nMN, title, graph, logger=None):
    import numpy as np
    import scipy as sp

    #Description
    mse = np.sum(np.square(predictions - truth)) / nMN
    cor = np.corrcoef(predictions, truth)[0, 1]
    rho, _ = sp.stats.spearmanr(a=predictions, b=truth)

    output = logger if logger else print
    output('-' * 60)
    output(title)
    output('MSE = {}'.format(mse))
    output('Correlation = {}'.format(cor))
    output('RankCorrelation = {}'.format(rho))

    #Graphing
    if graph: scatter([truth, predictions], ['truth', 'predictions'])

    return mse, cor, rho


#--Acquire ids of a k-fold training testing set
def kFold(k, nMN, seed=1):
    import numpy as np

    #Reset the seed
    np.random.seed(seed=seed)

    #The indice to be selected
    rMN = np.arange(nMN)
    np.random.shuffle(rMN)

    #Indicator
    #To make sure the distribution is as evenly as possible
    ind = abs(nMN - (nMN // k + 1) * (k - 1) - (nMN // k + 1)) < abs(nMN - (nMN // k) * (k - 1) - (nMN // k))

    #Series id based on k
    anchor = np.arange(k) * (nMN // k + ind)
    
    #Acquire the training and testing set ids
    id_test = [rMN[anchor[i]:(anchor[i + 1] if i + 1 != len(anchor) else None)] for i in range(len(anchor))]
    id_train = [np.setdiff1d(rMN, id_test[i]) for i in range(len(id_test))]

    #Deal with 1 fold (using entire dataset to train and test)
    if k == 1: id_train = id_test

    return id_train, id_test
    

#--Logger
def iniLogger(loggerName, fileName, _console):
    import logging

    #Use the default logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)

    #Create formatter
    formatter = logging.Formatter('%(message)s')

    #Create file handler and add to logger
    fh = logging.FileHandler('../log/{}'.format(fileName), mode='w+')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    #Create console handler and add to logger
    if _console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


#--Get the closest N neighbors
#Input: m by k (dimension) numpy array
#Output: top N neighbor ids
def getClosestNeighbors(topN, data, metric='euclidean'):
    import numpy as np
    import scipy as sp

    distance = sp.spatial.distance.squareform(sp.spatial.distance.pdist(data, metric))
    neighborIds = np.argsort(distance, axis=-1)[:, 1:topN + 1]

    return neighborIds