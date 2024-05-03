import numpy as np


class Linear:

    @classmethod
    def calcular(cls, X):
        """f(x) = x"""
        return X
    
    @classmethod
    def derivada(cls, X):
        """f'(x) = 1"""
        return np.ones_like(X)