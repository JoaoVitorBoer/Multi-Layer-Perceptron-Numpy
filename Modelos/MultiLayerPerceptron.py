import numpy as np
import matplotlib.pyplot as plt

class MLP:

    def __init__(self, layers, otimizador, loss):
        self.layers = layers
        self.epocas = None
        self.otimizador = otimizador
        self.loss = loss
        self.save_loss = []
        self.save_acuracia = []
        self.save_R2 = []

        self.gradientes = []
        self.pesos = []
        """Inicializa as dimensões de entrada das camadas"""
        for i in range(1, len(self.layers)):
            self.layers[i].input_dim = self.layers[i-1].n_units
        
        """Inicializa os pesos das camadas"""
        for layer in self.layers:
            layer.pesos = layer.inicializador.inicializar(layer.input_dim, layer.n_units)

    def calcular_acuracia(self, y, y_chapeu):
        return np.mean(np.argmax(y, axis=1) == np.argmax(y_chapeu, axis=1))

    def calcular_R2(self, y, y_chapeu):
        y_media = np.mean(y)
        ss_total = np.sum((y - y_media)**2)
        ss_res = np.sum((y - y_chapeu)**2)
        r_quadrado = 1 - (ss_res / ss_total)
        return r_quadrado
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, gradiente_loss):
        dl_dy_anterior = gradiente_loss
        """derivada da loss em relação ao output da ultima camada"""
        for layer in reversed(self.layers):
            dl_dy_anterior = layer.backward(dl_dy_anterior)

    def weights_hist(self, axs, epoch):
        for layer in range(len(self.layers)):
            axs[epoch, layer].hist(self.layers[layer].pesos.flatten())
            axs[epoch, layer].set_title(f'Layer {layer} - Epoch {epoch}')
            axs[epoch, layer].set_ylabel('Count')
            axs[epoch, layer].set_xlabel('Weigths')

        return None

    def gradient_hist(self, axs, epoch): 
        for layer in range(len(self.layers)):
            axs[epoch, layer].hist(self.layers[layer].gradiente_pesos.flatten())
            axs[epoch, layer].set_title(f'Layer {layer} - Epoch {epoch}')
            axs[epoch, layer].set_ylabel('Count')
            axs[epoch, layer].set_xlabel('Gradients')

        return None
    
    def fit(self, X, y, epocas, report=False):
        self.epocas = epocas

        if report:
            fig_weights_hist = plt.figure(figsize=(2.5 * 3, 2.5 * - ( - self.epocas * len(self.layers)) / 4), constrained_layout = True)
            fig_weights_hist.suptitle('Weights Histogram w.r.t Epochs')
            gs_weights_hist = fig_weights_hist.add_gridspec(self.epocas, len(self.layers))
            axs_weights_hist = gs_weights_hist.subplots()
    

            fig_gradient_hist = plt.figure(figsize=(2.5 * 3, 2.5 * - ( - self.epocas * len(self.layers)) / 4), constrained_layout = True)
            fig_gradient_hist.suptitle('Gradient Histogram w.r.t Epochs')
            gs_gradient_hist = fig_gradient_hist.add_gridspec(self.epocas, len(self.layers))
            axs_gradient_hist = gs_gradient_hist.subplots()
        
        for i in range(self.epocas):
           
            y_chapeu = self.forward(X)
            perda = self.loss.calcular(y, y_chapeu)

            # Gradiente - Derivada parcial da função de custo em relação à saída da rede
            dl_dy = self.loss.gradiente(y, y_chapeu)

            # Computa o gradiente para as camadas ocultas
            self.backward(dl_dy)
            self.otimizador.atualizar_pesos(self.layers)
            
            R2 = self.calcular_R2(y, y_chapeu)
            acuracia = self.calcular_acuracia(y, y_chapeu)
            media_loss = np.mean(perda)
            print(f"Epoch: {i+1}/{self.epocas} - loss: {media_loss:.6f} - acuracia: {acuracia:.6f} - R²: {R2:.6f}")

            self.save_R2.append(R2)
            self.save_loss.append(media_loss)
            self.save_acuracia.append(acuracia)
            if report:
                self.gradient_hist(axs_gradient_hist, i)
                self.weights_hist(axs_weights_hist, i)
        
        if report:

            fig_weights_hist.savefig('./Graficos/WH.png')
            fig_gradient_hist.savefig('./Graficos/GH.png')

            self.plot_histograms()


    def plot_histograms(self):
        plt.figure(figsize=(15, 6))
        for i, layer in enumerate(self.layers):
            # Ativações
            activations = np.concatenate(layer.save_ativacoes, axis=0)
            plt.subplot(1, len(self.layers) * 2, 2*i+1)
            plt.hist(activations.ravel(), bins=50)
            plt.title(f'Ativações - Camada {i+1}')
            # Gradientes
            gradients = np.concatenate([g.ravel() for g in layer.save_gradientes], axis=0)
            plt.subplot(1, len(self.layers) * 2, 2*i+2)
            plt.hist(gradients, bins=50)
            plt.title(f'Gradientes - Camada {i+1}')
        plt.tight_layout()
        plt.savefig('Graficos/histograms.png')

    def predict(self, X):
        return self.forward(X)


    def plot_loss_epochs(self):
        plt.figure()
        plt.plot(self.save_loss, label='Loss')
        plt.title('Loss vs. Epocas')
        plt.xlabel('Epocas')
        plt.ylabel('Value')
        plt.savefig('Graficos/loss_epochs.png')
    
    def plot_acuracy_epoch(self):
        plt.figure()
        plt.plot(self.save_acuracia, label='acuracia')
        plt.title('acuracia vs. Epocas')
        plt.xlabel('Epocas')
        plt.ylabel('Value')
        plt.savefig('Graficos/acuracia_epochs.png')
        
    def plot_R2_epoch(self):
        plt.figure()
        plt.plot(self.save_R2, label='R2')
        plt.title('R2 vs. Epocas')
        plt.xlabel('Epocas')
        plt.ylabel('Value')
        plt.savefig('Graficos/R2_epochs.png')    
        
