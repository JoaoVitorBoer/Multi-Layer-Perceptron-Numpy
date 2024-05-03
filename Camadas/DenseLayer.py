import numpy as np

class Dense:

    def __init__(self, n_units, ativacao, inicializador, use_bias, input_dim=None):
        """
        n_units: nº de neurônios da camada atual
        ativacao: Classe da função de ativação
        inicializador: Classe do inicializador
        bias: Se a camada terá um viés
        (opcional) input_dim: nº de input features para a camada
        """
        self.input_dim = input_dim
        self.n_units = n_units
        self.ativacao = ativacao
        self.inicializador = inicializador

        self.save_ativacoes = []  # Lista para armazenar ativações
        self.save_gradientes = []    # Lista para armazenar gradientes dos pesos
        self.delta = None
        self.pesos = None
        self.pre_ativacao = None
        self.X = None
        self.use_bias = use_bias
        if use_bias:
            self.bias = np.zeros((1, n_units))


    def forward(self, x):
        self.X = x

        if self.use_bias:
            self.pre_ativacao = np.dot(self.X, self.pesos) + self.bias
            
        else:
            self.pre_ativacao = np.dot(self.X, self.pesos)

        s = self.ativacao.calcular(self.pre_ativacao)
        self.save_ativacoes.append(s)
        return s
    
    def backward(self, gradiente_saida_anterior):
        
        # Derivada da funcao de ativacao 
        derivada_ativacao = self.ativacao.derivada(self.pre_ativacao)

       
        self.delta = np.multiply(gradiente_saida_anterior, derivada_ativacao)

        self.gradiente_pesos = np.matmul(self.X.T, self.delta)
        self.save_gradientes.append(self.gradiente_pesos)
        
        if self.use_bias:
            self.gradiente_bias = np.sum(self.delta, axis=0, keepdims=True)
        
        # Calcula o gradiente do erro para propagar para a camada anterior
        dl_dy_novo = np.matmul(self.delta, self.pesos.T)
    
        return dl_dy_novo