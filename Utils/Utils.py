

class Utils:

    @classmethod
    def print_model_info(cls, model):
        print(f"""MLP: 
            {len(model.layers)} layers
            Learning_rate: {model.otimizador.learning_rate}
            Otimizador: {model.otimizador}
            Loss: {model.loss.__name__}\n""")

        print(' '*12 + '-'*20 + ' Layers ' + '-'*20)
        for i in range(len(model.layers)):
            output = ' '*12 + f'Layer {i}:  Input_dim: {model.layers[i].input_dim}     Units: {model.layers[i].n_units}     Shape Weights: {model.layers[i].pesos.shape}      Ativacao: {model.layers[i].ativacao.__name__}      Inicializador: {model.layers[i].inicializador.__name__}'
            print(output)

        print('\n')
    

