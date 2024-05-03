import numpy as np


class Relu:
    
    @classmethod
    def calcular(cls, X):
        """max(0, X) -> Se X > 0, retorna X, senão, retorna 0"""
        return np.maximum(X, 0)
    
    @classmethod
    def derivada(cls, X):
        """Se X > 0, retorna 1, senão, retorna 0"""
        return np.where(X > 0, 1,0)