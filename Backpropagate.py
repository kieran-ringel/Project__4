import numpy as np
class Backpropagate:
    def __init__(self):
        pass
    def backpropagate(self, deltas, node_values, trainpoints):
        '''Kieran Ringel
        Over arching function to backpropagate the error that has been calculated for each node
        Used to allow momentum to occur and tune learning rate'''
        eta = .07  # learning rate TUNE
        mf = 14  # momentum factor TUNE
        # gets the delta w value for each weight
        change = Backpropagate.deltaW(self, eta, deltas, node_values, trainpoints)
        for layer in range(len(self.NN)):
            for node in range(len(self.NN[layer])):
                for weight in range(len(self.NN[layer][node])):
                    if self.pastError == None:  # if it is the first data point, momentum cannot be used
                        self.NN[layer][node][weight] += change[layer][node][weight]
                    else:  # applies momentum
                        self.NN[layer][node][weight] += change[layer][node][weight] + (
                                    mf * self.pastError[layer][node][weight])
        self.pastError = change  # saves current error as previous before current is recalculated


    def deltaW(self, learn_rate, deltas, node_values, trainpoints):
        '''Kieran Ringel
        Calculated delta w as the negation of the learning rate times the partial derivative of the error
        with respect to the weights
        Saves delta w's for each weight in an array to be used in backpropagate'''
        change = []
        for layer in reversed(range(len(self.NN))):
            layernode_change = []
            if layer != 0:  # if not the input layer
                for node in range(len(self.NN[layer])):
                    node_change = []
                    for inputnode in range(len(node_values[layer - 1])):
                        # multiplies learning rate by error on the node holding that weight by the node value of the previous node
                        weight_change = learn_rate * deltas[layer][node] * node_values[layer - 1][inputnode]
                        node_change.append(weight_change)
                    node_change.append(learn_rate * deltas[layer][node])  # for bias node, since input value would be a 1
                    layernode_change.append(node_change)
                change.insert(0, layernode_change)  # inserts at front of list to keep in order with 'reverse' above
            if layer == 0:  # if it is the input layer so that values of node values of previous node are the training points
                for node in range(len(self.NN[layer])):
                    node_change = []
                    for inputnode in range(len(trainpoints[:-1])):
                        # multiplies learning rate by error on the node holding that weight by the node value of the previous node, training values
                        weight_change = learn_rate * deltas[layer][node] * trainpoints[inputnode]
                        node_change.append((weight_change))
                    node_change.append(learn_rate * deltas[layer][node])  # for bias node, since input value would be a 1
                    layernode_change.append(node_change)
                change.insert(0, layernode_change)
        return (change)

    def backerror(self, output, expected):
        """Kieran Ringel
        Back propagates the error, returns a matrix of the error on each node in the NN"""
        tot_error = 0
        errorarr = []
        for layer in reversed(range(len(output))):  # reversed for backpropagation
            layererror = []
            if output[-1] == output[layer]:  # if output layer, calculate error of output nodes
                nodeerror = []
                for node in range(len(output[layer])):
                    if self.classification == 'classification':  # iterates over all output nodes
                        outputindex = output[layer].index(output[layer][node])  # gets index of output node
                        inputindex = self.classes.index(expected)  # gets index of expected output
                        if outputindex == inputindex:  # if they are the same the expected probability is 1
                            expected_val = 1
                        if outputindex != inputindex:  # if they are the same the expected probability is 0
                            expected_val = 0
                    if self.classification == 'regression':  # for regression will only go over one output node
                        expected_val = float(expected)
                    error = expected_val - output[layer][node]  # error is expected minus calculated
                    tot_error += error
                    nodeerror.append(error)
            else:  # if hidden layer use delta rule to calculate delta or error of hidden nodes
                for node in range(len(output[layer])):
                    error = 0
                    for errornode in range(len(errorarr[layer - 1])):
                        for weight in range(len(self.NN[layer][node])):
                            error += self.NN[layer][node][weight] * errorarr[layer - 1][
                                errornode]  # summation of weight between node and node of next layer multiplied by error of next node
                        nodeerror.append(error)
            for node in range(len(output[layer])):
                newerror = nodeerror[node] * Backpropagate.derivative(self,
                    output[layer][node])  # muliply all errors by the derivative of the activation function
                layererror.append(newerror)
            errorarr.insert(0, layererror)
        return (errorarr)


    def derivative(self, output):
        '''Kieran Ringel
        Derivative of the activation function'''
        if self.classification == 'classification':
            return output * (1 - output)
        if self.classification == 'regression':
            return 1
