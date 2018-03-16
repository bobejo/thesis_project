import keras.backend
"Will be removed when other network is created"

def logloss( y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1)

def accuracy(y_true, y_pred):

    true_label = keras.backend.round(y_true)
    pred_label = keras.backend.round(y_pred)

    return keras.backend.mean(keras.backend.equal(true_label, pred_label))

class LogLoss:
    def __init__(self):
        self.__name__ = 'LogRegLoss'
    #self.__name__ = 'LogRegLoss'

    def __call__(self, y_true, y_pred):
        return keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1)

