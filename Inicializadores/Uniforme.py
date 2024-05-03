import numpy as np


class Uniforme:
    
    @classmethod
    def inicializar(cls, input_dim, n_units):
        limite_inferior = -1
        limite_superior = 1
        return np.random.uniform(limite_inferior, limite_superior, (input_dim, n_units))