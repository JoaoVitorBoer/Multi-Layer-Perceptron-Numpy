import numpy as np

class MeanSquaredError:

    @classmethod
    def calcular(cls, y, y_pred):
        return np.mean(np.square(y - y_pred))
    
    @classmethod
    def gradiente(cls, y, y_pred):
            N = y.shape[0]
            grad = (2 / N) * (y_pred - y)
            return grad
    
