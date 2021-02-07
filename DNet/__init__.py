from . import model
from . import optimizers
from . import layers

from .optimizers import SGD, RMSprop, Adam
from .model import NNet
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
