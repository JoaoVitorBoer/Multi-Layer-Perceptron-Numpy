
import numpy as np

class SoftMax:
    
    @classmethod
    def calcular(self, X):
        return np.exp(X)/np.sum(np.exp(X))

    @classmethod
    def derivada(self, X):
        return self.calcular(X) * (1 - self.calcular(X))
