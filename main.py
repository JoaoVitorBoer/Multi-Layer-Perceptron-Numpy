from Ativacoes.LeakyRelu import LeakyRelu
from Ativacoes.Relu import Relu
from Ativacoes.Sigmoid import Sigmoid
from Ativacoes.Tanh import Tanh
from Ativacoes.Linear import Linear
from Ativacoes.Softmax import SoftMax

from Inicializadores.HeNormal import HeNormal
from Inicializadores.XavierNormal import XavierNormal
from Inicializadores.Normal import Normal
from Inicializadores.Uniforme import Uniforme

from Datasets.Datasets import Datasets

from FuncoesDeCusto.BCE import BinaryCrossEntropy
from FuncoesDeCusto.MSE import MeanSquaredError
from FuncoesDeCusto.CCE import CategoricalyCrossEntropy

from Otimizadores.SGD import SGD
from Otimizadores.Adam import Adam

from Camadas.DenseLayer import Dense
from Modelos.MultiLayerPerceptron import MLP

from Utils.Utils import Utils
import warnings
import pickle

import numpy as np


warnings.filterwarnings(action="ignore")

x_train, y_train, x_test, y_test, input_size, num_labels, tipo = Datasets.load_fashion_mnist()

print('Starting model...\n')

print(f'input_size for the first layer: {input_size} \nnum_labels for the n_units output layer: {num_labels}\n')

layers = [Dense(n_units=100, ativacao=Relu, inicializador=HeNormal, input_dim=input_size, use_bias=True),
          Dense(n_units=100, ativacao=Relu, inicializador=HeNormal,  use_bias=True),
          Dense(n_units=num_labels, ativacao=Linear, inicializador=HeNormal,  use_bias=True)]



#otimizador = Adam()
otimizador = SGD(learning_rate=0.01)

model = MLP(layers=layers, otimizador=otimizador, loss=CategoricalyCrossEntropy)

Utils.print_model_info(model)


model.fit(X=x_train, y=y_train, epocas=20, report=False)
model.plot_loss_epochs()
model.plot_acuracy_epoch()
model.plot_R2_epoch()

dump = False
if dump:
    pickle.dump(model, open(f'ModelDumps/model.pkl', 'wb'))

y_pred = model.forward(x_test)

if tipo == 'binary':
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test) * 100
    print(f'Accuracy: {accuracy:.2f}%')

elif tipo == 'multiclass':
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred == y_test) * 100
    print(f'Accuracy: {accuracy:.2f}%')
