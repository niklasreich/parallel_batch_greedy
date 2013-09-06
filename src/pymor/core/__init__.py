# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import (BasicInterface, ImmutableInterface, abstractmethod, abstractclassmethod,
                                   abstractstaticmethod, abstractproperty, inject_sid)
from pymor.core.logger import getLogger

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

from functools import partial


class Unpicklable(object):
    '''Mix me into classes you know cannot be pickled.
    Our test system won't try to pickle me then
    '''
    pass


dump = partial(pickle.dump, protocol=-1)
dumps = partial(pickle.dumps, protocol=-1)
load = pickle.load
loads = pickle.loads
