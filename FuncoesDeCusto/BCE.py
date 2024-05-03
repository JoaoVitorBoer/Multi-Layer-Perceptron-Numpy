import numpy as np


class BinaryCrossEntropy:

    @classmethod
    def calcular(cls, y, y_pred):
        
        # Evita valores de log(0) resultando em 'nan'
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        bce = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return bce

    @classmethod
    def gradiente(cls, y, y_pred):
        # Evita valores de log(0) resultando em 'nan'
        epsilon = 1e-15

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        grad = (y_pred - y) / (y_pred * (1 - y_pred))
        return grad
