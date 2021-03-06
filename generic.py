# -- Decorators for functions
class FuncDecorator():

    def timeOperation(original_function):
        """
        Print elapsed time of 1 time operation.
        """
        def new_function(*args, **kwargs):
            import datetime

            before = datetime.datetime.now()
            original_output = original_function(*args, **kwargs)
            after = datetime.datetime.now()
            print('Elapsed Time = {0}'.format(after - before))

            return original_output
        return new_function

    def delayOperation(time):
        """
        Delay operation by `time` secs
        - 0.7*time + 0.6*random()*time
        - When time=10, it's 7-13 secs
        """
        from time import sleep
        from random import random
        def wrapper(original_function):

            def new_function(*args, **kwargs):
                sleep(0.7 * time + 0.6 * random() * time)

                original_output = original_function(*args, **kwargs)
                return original_output

            return new_function
        return wrapper


# -- Decorators for classes
class ClsDecorator():

    def prohibitAttrSetter(cls):
        """
        Prohibit access to attribute setter
        """
        def setattr(self, key, value):
            class ProhibittedOperation(Exception): pass
            raise ProhibittedOperation('Not allowed to modify attributes directly.')

        cls.__setattr__ = setattr
        return cls

    def grantKeywordUpdate(cls):
        """
        Grant attribute modification by `update` method
        """
        def update(self, **kwarg):
            for key, value in kwarg.items():
                self.__dict__[key] = value

        cls.update = update
        cls.__init__ = update
        return cls


# -- Universal container
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


# -- Setting container
@ClsDecorator.prohibitAttrSetter
@ClsDecorator.grantKeywordUpdate
class SettingContainer(UniversalContainer):
    """
    Usage
    - Convenient keyword = parameter setting.
    - Protected attribute setter. Use `update(key=value)` to modify content.
    """
    pass


# -- Convert data to object form (recursive)
# If wants to restrict access to attributes, inherit SettingContainer instead
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


# -- Import config files as an object
def getConfigObj(path):
    """
    Read config files into an obj container.
    - Support file type: .json, .yml.
    """
    import yaml
    import json
    import re

    class UnknownFileType(Exception): pass

    with open(path, 'r', encoding='utf-8') as f:
        ext = re.search('\.(.+)', path).group(1)
        if ext == 'json': config_dic = json.load(f)
        elif ext == 'yml': config_dic = yaml.load(f)
        else: raise UnknownFileType('\'{}\' is not a supported file type.'.format(ext))

    return ConvertedContainer(config_dic)


def writeJsls(obj, path):
    """
    Write all objs of a iterable into a jsl file
    """
    import json
    import numpy

    # Deal with the json default encoder defect
    # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, numpy.integer):
                return int(obj)
            elif isinstance(obj, numpy.floating):
                return float(obj)
            elif isinstance(obj, numpy.ndarray):
                return obj.tolist()
            else:
                return super(NumpyEncoder, self).default(obj)

    with open(path, mode='a') as f:
        for item in obj:
            json.dump(item, f, cls=NumpyEncoder)
            f.write('\n')

    print('Completed writing \'{}\', appended obj len {}.'.format(path, len(obj)))


def readJsls(path):
    """
    Read all objs in one jsl file
    """
    import json

    output = []
    with open(path, mode='r') as f:
        for line in f:
            output.append(json.loads(line))

    print('Completed reading \'{}\', loaded obj len {}.'.format(path, len(output)))
    return output


def flattenList(l, nLevel=-1):
    """
    Flatten list (recursive for `nLevel`)
    - Parameter: `l`, a list; `nLevel`=-1 if extract all levels
    - Return: a flattened list as a generator
    """
    import collections

    for el in l:
        if isinstance(el, collections.Sequence) and not isinstance(el, (str, bytes)) and nLevel != 0:
            yield from flattenList(el, nLevel - 1)
        else:
            yield el


# -- Element-wise list operation
# Return: operated list
def listEWiseOp(op, l):
    return list(map(op, l))


# -- Graphing 2-D scatter plot
# With distribution and linear fitting line
def scatter(vectors, names):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    sns.set(color_codes=True)

    df = pd.DataFrame({
        names[0]: vectors[0],
        names[1]: vectors[1]
    })

    sns.jointplot(x=names[0], y=names[1], data=df, color="m", kind="reg", scatter_kws={"s": 10})
    plt.show()


# -- Evaluate model with mse, cor, and graphing
def evalModel(predictions, truth, nMN, title, graph, logger=None):
    import numpy as np
    import scipy as sp

    # Description
    mse = np.sum(np.square(predictions - truth)) / nMN
    cor = np.corrcoef(predictions, truth)[0, 1]
    rho, _ = sp.stats.spearmanr(a=predictions, b=truth)

    output = logger if logger else print
    output('-' * 60)
    output(title)
    output('MSE = {}'.format(mse))
    output('Correlation = {}'.format(cor))
    output('RankCorrelation = {}'.format(rho))

    # Graphing
    if graph: scatter([truth, predictions], ['truth', 'predictions'])

    return mse, cor, rho


# -- Acquire ids of a k-fold training testing set
def kFold(k, nMN, seed=1):
    import numpy as np

    # Reset the seed
    np.random.seed(seed=seed)

    # The indice to be selected
    rMN = np.arange(nMN)
    np.random.shuffle(rMN)

    # Indicator
    # To make sure the distribution is as evenly as possible
    ind = abs(nMN - (nMN // k + 1) * (k - 1) - (nMN // k + 1)) < abs(nMN - (nMN // k) * (k - 1) - (nMN // k))

    # Series id based on k
    anchor = np.arange(k) * (nMN // k + ind)

    # Acquire the training and testing set ids
    id_test = [rMN[anchor[i]:(anchor[i + 1] if i + 1 != len(anchor) else None)] for i in range(len(anchor))]
    id_train = [np.setdiff1d(rMN, id_test[i]) for i in range(len(id_test))]

    # Deal with 1 fold (using entire dataset to train and test)
    if k == 1: id_train = id_test

    return id_train, id_test


# -- Logger
def initLogger(loggerName, console=True, consoleLevel='DEBUG', fileLevel='INFO'):
    """
    Initialize a logger using logging module
    - INFO or up will be saved in file.
    - DEBUG or up will be printed in console.
    - https://docs.python.org/3/library/logging.html#logging-levels.
    - More information is logged in log file than in console.
    """
    import logging

    # Create new logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)

    # Formatter reference
    # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create file handler and add to logger
    fh = logging.FileHandler('log/{}.log'.format(loggerName), mode='w+')
    fh.setLevel(fileLevel)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Create console handler and add to logger
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(consoleLevel)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    return logger


# -- Structured logging
class LogMsg():
    import json
    from typing import Callable, Any

    # Py3.8 new syntax for positional arg
    # def __init__(self, handler: str, /, **kwargs):
    def __init__(self, handler: Callable, **kwargs: Any):
        self.msg = {'handler': handler.__name__, **kwargs}

    def __str__(self) -> str:
        return self.json.dumps(self.msg)


class Logger():
    import logging
    from typing import Tuple, List, Union

    formatStr = '''{
        "name": "%(name)s",
        "level": "%(levelname)s",
        "time": "%(asctime)s",
        "lineUserId": "unknown",
        "message": %(message)s
    }'''
    logger: logging.Logger
    handlers: List[logging.Handler]

    def __init__(self, name, *handlerConfigs: Tuple[logging.Handler, int]):
        self.logger = self.logging.getLogger(name)
        for handlerConfig in handlerConfigs:
            self.addHandler(*handlerConfigs)

    def addHandler(self, handler: logging.Handler, handlerLevel: int):
        handler.setLevel(handlerLevel)
        handler.setFormatter(self.logging.Formatter(self.formatStr))
        self.logger.addHandler(handler)

    def setLineUserId(self, lineUserId: str):
        self.formatStr = self.formatStr.replace(
            '"lineUserId": "unknown"',
            '"lineUserId": "{}"'.format(lineUserId), 1
        )
        for handler in self.handlers:
            handler.setFormatter(self.logging.Formatter(self.formatStr))

    def error(self, msg: str):
        self.logger.error(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def info(self, msg: LogMsg):
        # Json format
        self.logger.info(msg)

    def debug(self, msg: Union[LogMsg, str]):
        self.logger.debug(msg)


# -- Make exception message in one line for better logging
def flattenErrorTraceback(e: BaseException) -> str:
    import traceback

    eTraceback = traceback.format_exception(value=e)
    eTraceback_ines = []
    for line in [line.rstrip('\n') for line in eTraceback]:
        eTraceback_ines.extend(line.splitlines())

    return eTraceback_ines


# -- Get the closest N neighbors
# Input: m by k (dimension) numpy array
# Output: top N neighbor ids
def getClosestNeighbors(topN, data, metric='euclidean'):
    import numpy as np
    import scipy as sp

    distance = sp.spatial.distance.squareform(sp.spatial.distance.pdist(data, metric))
    neighborIds = np.argsort(distance, axis=-1)[:, 1:topN + 1]

    return neighborIds


def divideSets(proportion, nSample, seed=1):
    """
    Acquire ids of arbitrary set division
    - Given proportion and the number of samples, return ids (starts from 0) of each set [set, set, ...].
    - An element in `proportion` represents a set.
    - The proportion must sum to 1.
    """
    import numpy as np

    assert sum(proportion) == 1, '`proportion` must sum to 1.'

    # Reset np seed
    np.random.seed(seed=seed)

    # Number of indice for each set
    nIds = np.around(np.array(proportion) * nSample)

    # Shuffle the indice pool
    rIndiceGen = np.arange(nSample)
    np.random.shuffle(rIndiceGen)
    rIndiceGen = (i for i in rIndiceGen)

    # Assign indice to each set
    ids = []
    for nId in nIds:
        id = []
        while nId > 0:
            try: id.append(next(rIndiceGen))
            except StopIteration: break
            nId -= 1
        ids.append(id)

    return ids
