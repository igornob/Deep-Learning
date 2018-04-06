from numpy import exp, array, random, dot
import json

class NeuronLayer():

    """
        Classe que representa cada camada oculta
            Inicia os pesos aleatoriamente
    """

    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):

        print("\n\t TREINO ESTOCASTICO\n")

        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment



    def train_lote(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        
        print("\n\t TREINO POR LOTE\n")

        layer1_weight_offset = self.layer1.synaptic_weights
        layer2_weight_offset = self.layer2.synaptic_weights

        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            layer1_weight_offset += layer1_adjustment
            layer2_weight_offset += layer2_adjustment


        self.layer1.synaptic_weights = layer1_weight_offset
        self.layer2.synaptic_weights = layer2_weight_offset

    def train_mini_lote(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        
        print("\n\t TREINO POR MINI LOTE\n")

        layer1_weight_offset = self.layer1.synaptic_weights
        layer2_weight_offset = self.layer2.synaptic_weights

        iteracao = 0
        BATCH_SIZE = 10

        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            layer1_weight_offset += layer1_adjustment
            layer2_weight_offset += layer2_adjustment

            if iteracao % BATCH_SIZE == 0:

                #print("Mini bath atualizando...", iteracao)
                self.layer1.synaptic_weights = layer1_weight_offset
                self.layer2.synaptic_weights = layer2_weight_offset

                layer1_weight_offset = self.layer1.synaptic_weights
                layer2_weight_offset = self.layer2.synaptic_weights


            iteracao+=1

    def train_termo_momento(self, training_set_inputs, training_set_outputs, number_of_training_iterations):

        print("\n\t TREINO TERMO DO MOMENTO\n")

        MOMENTO = 0.3
        
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            current_layer1_w = self.layer1.synaptic_weights
            current_layer2_w = self.layer2.synaptic_weights
            
            # Adjust the weights.
            current_layer1_w += layer1_adjustment
            current_layer2_w += layer2_adjustment

            layer1_variation = self.layer1.synaptic_weights - current_layer1_w
            layer2_variation = self.layer2.synaptic_weights - current_layer2_w

            self.layer1.synaptic_weights += MOMENTO * layer1_variation + layer1_adjustment
            self.layer2.synaptic_weights += MOMENTO * layer2_variation + layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print ("    Layer 1 ")
        print (self.layer1.synaptic_weights)
        print ("    Layer 2")
        print (self.layer2.synaptic_weights)

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)


    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    #                                 0.1        0.2        0.3        0.4        0.5        0.6        0.7        0.8
    
    #training_set_inputs = array([ [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1] ])

    #training_set_outputs = array([ [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] ]).T
    
    #print("output original", training_set_outputs)



    #XOR

    #training_set_inputs = array([ [0.1, 0.1], [-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1], [0.6, 0.6], [-0.6, 0.6], [-0.6, -0.6], [0.6, -0.6] ])

    #training_set_outputs = array([ [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] ]).T
    

    # fim xor

    # INICIA input questao 3b
    
    with open('data_fx.json', 'r') as f:
	json_data = json.load(f)    
    
   
    input_offset = []
    output_offset = []

    #print(json_data)
    for el in json_data:
        

        output_el = [ el["fx"] ]
	input_el = [ el["x"] ]
        input_offset.append(input_el)
        output_offset.append(output_el)

    #print("input_offset", input_offset)

    #print("output_offset", output_offset)
    #exit()

    training_set_inputs = array(input_offset)
    training_set_outputs = array(output_offset).T
   
    # FIM input questao 3b


    
    """
        Classe que representa cada camada oculta
            Inicia os pesos aleatoriamente


    # INICIA input questao 4
    
    with open('data.json', 'r') as f:
        json_data = json.load(f)    
    
   
    input_offset = []
    output_offset = []

    #converte classe => numero
    class_convertion = { "c1": 0.1, "c2": 0.2, "c3": 0.3, "c4": 0.4, "c5": 0.5, "c6": 0.6, "c7": 0.7, "c8": 0.8}
    
    #print(json_data)
    for x in json_data:
        
        #print(x)
        #print(json_data[x])
        
        for el in json_data[x]:

            input_el = [ el["x"], el["y"] ]

            input_offset.append(  input_el )
            output_offset.append( class_convertion[x] )

    #print("input_offset", input_offset)

    #print("output_offset", output_offset)
    #exit()

    training_set_inputs = array( input_offset )
    training_set_outputs = array([output_offset]).T
   
    # FIM input questao 4 
	"""

    # Numero de W da primeira camada, deve ser igual ao tamanho da entrada
    num_w = len(training_set_inputs[0])

    # Numero de neuronios da primeira camada oculta, deve ser igual ao NUMERO DE W da segunda camada
    num_neur_cam_oculta = 4

    #print(num_neur_cam_oculta,"num_neur_cam_oculta")
    #exit()
    layer1 = NeuronLayer(num_neur_cam_oculta, num_w)

    # 1 = NEURONIO UNICO DE SAIDA (PERIGOSO DE MEXER)
    layer2 = NeuronLayer(1, num_neur_cam_oculta)

    # Combine the layers to create a neural network
    # Combina as camadas numa classe e organiza tudo (faz nd pratico na real)
    neural_network = NeuralNetwork(layer1, layer2)

    print ("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # numero de iteracoes do loop principal
    NUM_EPOCH = 10

    # Treino estocastico (normal)
    neural_network.train(training_set_inputs, training_set_outputs, NUM_EPOCH)


    # treino em lote, so atualiza os pesos ao final de tds as NUM_EPOCH
    #neural_network.train_lote(training_set_inputs, training_set_outputs, NUM_EPOCH)


    # Mini lote, atualiza os pesos a cada X iteracoes
    #neural_network.train_mini_lote(training_set_inputs, training_set_outputs, NUM_EPOCH)

    #
    #neural_network.train_termo_momento(training_set_inputs, training_set_outputs, NUM_EPOCH)   

    # SO imprime os pesos APOS o treino
    print ("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Testa pra novos elementos, usando os pesos ja treinados
    
    #exemplo questao 1
    #print ("Stage 3) Considering a new situation [0.1, 0.1, 0.1] -> ?: ")
    #hidden_state, output = neural_network.think(array([0.1, 0.1, 0.1]))

	# exemplo questao 3b
    print ("Stage 3) Considering a new situation [0.1, 0.1] -> ?: ")
    hidden_state, output = neural_network.think(array([2.5, 0.1]))

	
    # exemplo questao 4
    #print ("Stage 3) Considering a new situation [0.1, 0.1] -> ?: ")
    #hidden_state, output = neural_network.think(array([0.1, 0.1]))


    print (output)





