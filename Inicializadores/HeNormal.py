import numpy as np

class HeNormal:
 
    @classmethod
    def inicializar(cls, input_dim, n_units):
        dp = np.sqrt(2.0 / input_dim)
        return np.random.normal(0.0, dp, (input_dim, n_units))