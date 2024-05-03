import numpy as np


class Sigmoid:
    
    @classmethod
    def calcular(cls, X):
        """1 /(1 + e^(-X))"""
        return 1 /(1 + np.exp(-X))
 
    @classmethod
    def derivada(cls, X):
        """sigmoid(x) * (1 - sigmoid(x))"""
        return cls.calcular(X) * (1.0 - cls.calcular(X))
    
   