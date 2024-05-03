import numpy as np


class Adam:


    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07) -> None:

            assert beta1 >= 0 and beta1 < 1, f'Beta 1 should be [0, 1). The current value is {beta1}'
            assert beta2 >= 0 and beta2 < 1, f'Beta 2 should be [0, 1). The current value is {beta2}'
            self.learning_rate = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.t = 0

            self.m = None
            self.v = None

    def atualizar_pesos(self, layers):
        #variáveis self.m e self.v devem ser inicializadas para combinar as dimensões dos pesos que serão atualizados
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(layer.gradiente_pesos) for layer in layers]
            self.v = [np.zeros_like(layer.gradiente_pesos) for layer in layers]

        self.t += 1
        
        for i, layer in enumerate(layers):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.gradiente_pesos
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.power(layer.gradiente_pesos, 2)

            m_hat = self.m[i] / (1 - np.power(self.beta1, self.t))
            
            v_hat = self.v[i] / (1 - np.power(self.beta2, self.t))

            layer.pesos -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))

        # self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        # self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(gradient, 2)

        # m_hat = self.m / (1 - np.power(self.beta1, self.t))
        # v_hat =  -> self.m / (1 - np.power(self.beta2, self.t))
            layer.pesos -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))

    
    def __str__(self) -> str:
        return 'Adam'