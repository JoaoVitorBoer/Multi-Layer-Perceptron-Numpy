import numpy as np


class LeakyRelu:

    @classmethod
    def calcular(cls, X, alpha=0.01):
        """Se X > 0, retorna X, senão, retorna X * alpha"""
        return np.where(X > 0, X, alpha * X)
    
    @classmethod
    def derivada(cls, X, alpha=0.01):
        """Se X > 0, retorna 1, senão, retorna alpha"""
        return np.where(X > 0, 1, alpha)