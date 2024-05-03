import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist
import numpy as np

class Datasets:

    @classmethod
    def load_mnist_binary(cls):
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        
        # Pegando somente 0 e 1 para usar o BCE
        train_mask = (y_train == 0) | (y_train == 1)
        test_mask = (y_test == 0) | (y_test == 1)

        x_train, y_train = x_train[train_mask], y_train[train_mask]
        x_test, y_test = x_test[test_mask], y_test[test_mask]

        num_labels = len(np.unique(y_train))


        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        image_size = x_train.shape[1]
        input_size = image_size**2

        x_train = np.reshape(x_train, [-1, input_size])
        x_train = x_train / 255

        x_test = np.reshape(x_test, [-1, input_size])
        x_test = x_test / 255

        return x_train, y_train, x_test, y_test, input_size, num_labels, 'binary'

    @classmethod
    def load_mnist(cls):
        (x_train, y_train),(x_test, y_test) = mnist.load_data()

        num_labels = len(np.unique(y_train))

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        image_size = x_train.shape[1]
        input_size = image_size**2

        x_train = np.reshape(x_train, [-1, input_size])
        x_train = x_train / 255

        x_test = np.reshape(x_test, [-1, input_size])
        x_test = x_test / 255

        return x_train, y_train, x_test, y_test, input_size, num_labels, 'multiclass'
    
    @classmethod
    def load_iris_dataset(cls):
        iris = load_iris()
        X = np.array(iris.data)
        y = np.array(iris.target)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        input_size = 4  # Número de características (features)
       
        num_labels = 2  # Número de classes
        
        print()
        return x_train, y_train, x_test, y_test, input_size, num_labels, 'binary'
    
    @classmethod
    def load_fashion_mnist(cls):
        (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

        num_labels = len(np.unique(y_train))

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        image_size = x_train.shape[1]
        input_size = image_size**2

        x_train = np.reshape(x_train, [-1, input_size])
        x_train = x_train / 255

        x_test = np.reshape(x_test, [-1, input_size])
        x_test = x_test / 255
      
        return x_train, y_train, x_test, y_test, input_size, num_labels, 'multiclass'