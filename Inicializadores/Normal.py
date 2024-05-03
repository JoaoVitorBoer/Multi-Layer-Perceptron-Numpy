import numpy as numpy

class Normal:

    @classmethod
    def inicializar(cls, input_dim, n_units):
        """ Inicializa os pesos da camada com uma distribuição normal com média 0 e desvio padrão 1. 
        Parâmetros: 
            input_dim: int : Dimensão da camada de entrada.
            n_units: int : Número de neurônios na camada.
        
        Inicializa os pesos da camada com uma distribuição normal com média 0 e desvio padrão 1.
        """

        media = 0.0
        dp = 1.0
        return numpy.random.normal(media, dp, (input_dim, n_units))