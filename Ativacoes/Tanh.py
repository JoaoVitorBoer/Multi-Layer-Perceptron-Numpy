import numpy as np

class Tanh:
    
    @classmethod
    def calcular(cls, X):
        """(e^x - e^-x) / (e^x + e^-x)"""
        # np.tanh(X)
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
    
    @classmethod
    def derivada(cls, X):
        """1 - tanh(x)^2"""
        return 1 - np.power(cls.calcular(X), 2)