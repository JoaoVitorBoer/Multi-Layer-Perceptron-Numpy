import numpy as np
from Ativacoes.Softmax import SoftMax
class CategoricalyCrossEntropy:

    @classmethod
    def calcular(cls, y, y_pred):
        epsilon = 1e-15 
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  
        y_pred = SoftMax.calcular(y_pred)
        cce = -np.sum(y * np.log(y_pred))

        return cce

    @classmethod
    def gradiente(cls, y, y_pred):
        # epsilon = 1e-15 
        # y_pred = np.clip(y_pred, epsilon, 1 - epsilon) 
        # grad = -y / y_pred
        # return grad / y_pred.shape[0]
        return y_pred - y
