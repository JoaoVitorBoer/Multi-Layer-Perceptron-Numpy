import numpy as np


class XavierNormal:

    @classmethod
    def inicializar(cls, input_dim, n_units):
        """ Inicializa os pesos da camada com Xavier Normal. 
        Parâmetros: 
            input_dim: int : Dimensão da camada de entrada.
            n_units: int : Número de neurônios na camada.
        
        Inicializa os pesos da camada com uma distribuição normal com média 0 e desvio padrão 1/sqrt(input_dim).
        """

        dp = np.sqrt(1.0 / input_dim)
        return np.random.normal(0.0, dp, (input_dim, n_units))