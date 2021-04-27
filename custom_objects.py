from keras import layers
from keras import backend

def swish(x):
    return (backend.sigmoid(x) * x)

class Swish(layers.Activation):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(swish, **kwargs)
        self.__name__ = 'swish'

