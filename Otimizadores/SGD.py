import numpy as np


class SGD:
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def __str__(self) -> str:
        return 'SGD'
    
    def atualizar_pesos(self, layers):
        for layer in layers:
            layer.pesos -= self.learning_rate * layer.gradiente_pesos
            if layer.use_bias:
                layer.bias -= self.learning_rate * layer.gradiente_bias