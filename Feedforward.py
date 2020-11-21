import numpy as np

class Feedforward:
    def __init__(self, hlayers, classification):
        self.hlayers = hlayers
        self.classification = classification

    def feedforward(self, trainpoint):
        print('Got to FF')
        '''Kieran Ringel
        Feeds fowards data by getting the dot product of the input values and the weights on those input values
        That dot product is multipled by the activation function
        that resulting node value is then 'fed forward' through the network until the last layer'''
        outputarray = []
        for layer in range(self.hlayers + 1):  # iterate through layers hlayers + 1 for output layer
            if layer == 0:
                node_vals = list(trainpoint[:-1])  # if looking at first layer, the input is the training data, not including the classification
            else:
                node_vals = new_node_vals  # otherwise the inputs are the outputs of the last layer
            new_node_vals = []
            print("Input", node_vals)
            for node in range(len(self.NN[layer])):  # for each node in the layer
                # gets dot product of weights and node values and then adds the bias node
                cur_node = np.dot(self.NN[layer][node][:-1], node_vals[:]) + self.NN[layer][node][-1]
                new_node_vals.append(cur_node)  # each current node value is appnded to list of new node values
            new_node_vals = Feedforward.activation(self, new_node_vals)  # node values are activated using activation function
            outputarray.append(list(new_node_vals))  # creates an output array with the values of every node
        if self.classification == 'classification':  # softmax activation is used
            output = Feedforward.softmax(self, new_node_vals)
            outputarray[-1] = output
        if self.classification == 'regression':
            outputarray[-1] = new_node_vals
        print("Output Node Values", outputarray)
        return (outputarray)

    def activation(self, dot):
        '''Kieran Ringel
        Returns the activation function depending on what type is being used
        Logistic sigmoidal for classification
        Linear for regresssion'''
        if self.classification == "classification":  # logistic sigmoid activation function
            neg = np.negative(dot).astype('float128')
            val = 1 / (1 + np.exp(neg))
            return val
        if self.classification == 'regression':  # linear activation function
            return dot

    def softmax(self, new_node_vals):
        '''Kieran Ringel
        Calculates the soft max so that the output layers for classificaiton are probabilites'''
        exp = np.exp(new_node_vals)
        prob = []
        for outputs in exp:
            prob.append(outputs / sum(exp))
        return (prob)