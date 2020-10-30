import pandas as pd
import numpy as np
import math

from NN import NN
class NeuralNet:
    def __init__(self, file, hlayers, hnodes, classification):
        print('hidden layers', hlayers)
        print('hidden nodes', hnodes)
        self.file = file
        self.hlayers = hlayers
        self.hnodes = hnodes
        self.pastError = None
        self.classification = classification
        self.initialNN = self.initNN(file, hlayers, hnodes, classification)
        self.tenfold(file)


    def initNN(self, file, hlayers, hnodes, classification):
        input = file.shape[1] - 1
        if classification == 'regression':
            print('init')
            output = 1
        if classification == 'classification':
            self.classes = list(file['class'].unique())
            output = file['class'].nunique()
        neuralNet = NN.getNN(self, input, hlayers, hnodes, output)
        return(neuralNet)

    def tenfold(self, file):
        # 10 fold cross validation by getting every 10th data point of the sorted data
        avgerror = 0
        fold = [None] * 10
        for cv in range(10):
            to_test = file.iloc[cv::10]
            fold[cv] = to_test

        for foldnum in range(10):  # get test and train datasets
            print("Run", foldnum+1)  #prints run number
            test = fold[foldnum]     #gets test set
            train_list = fold[:foldnum] + fold[foldnum+1:]    #gets training set, everything besided the test set
            train = pd.concat(train_list)           #concatanates the 2 parts of the test set
            train.reset_index(drop=True, inplace=True)  #resets index on both
            test.reset_index(drop=True, inplace=True)
            self.train(train)
            avgerror += self.test(test)
        print('Average Error', avgerror/10)

    def train(self, train):
        epochs = 1     #TUNE
        print('epochs', epochs)
        self.NN = self.initialNN
        for i in range(epochs):
            train = train.sample(frac=1).reset_index(drop=True)
            for row, trainpoints in train.iterrows():    #iterate through training data points
               node_values = self.feedforward(trainpoints)
               error = self.backerror(node_values, trainpoints['class'])
               self.backpropagate(error, node_values, trainpoints)

    def test(self, test):
        tot_error = 0
        for row, testpoints in test.iterrows():
            node_values = self.feedforward(testpoints)
            tot_error += self.calcerror(node_values[-1], testpoints['class'])
        print(tot_error/len(test))
        return(tot_error/len(test))

    def calcerror(self, output, expected):
        error = 0
        if self.classification == 'classification':
            for node in range(len(output)):
                outputindex = output.index(output[node])
                inputindex = self.classes.index(expected)
                if outputindex == inputindex:
                    rt = 1
                if outputindex != inputindex:
                    rt = 0
                error -= (rt * math.log10(output[node])) #+ ((1 - rt) * math.log(1 - output[node]))
        if self.classification == 'regression':
            error = ((float(output[0]) - float(expected)) ** 2) / 2      #I BELIVE MAYBE NOT
        return(error)

    def feedforward(self, trainpoint):
        outputarray = []
        for layer in range(self.hlayers + 1):  # iterate through layers
            if layer == 0:
                node_vals = list(trainpoint[:-1])
            else:
                node_vals = new_node_vals
            new_node_vals = []
            for node in range(len(self.NN[layer])):
                cur_node = np.dot(self.NN[layer][node][:-1], node_vals[:]) + self.NN[layer][node][-1]
                new_node_vals.append(cur_node)
            new_node_vals = self.activation(new_node_vals)
            outputarray.append(list(new_node_vals))
        if self.classification == 'classification':
            output = self.softmax(new_node_vals)
            outputarray[-1] = output
        if self.classification == 'regression':
            output = new_node_vals
            outputarray[-1] = output
        return(outputarray)

    def backpropagate(self, deltas, node_values, trainpoints):
        eta = .000001        #learning rate
        mf = 0         #momentum factor
        change = self.deltaW(eta, deltas, node_values, trainpoints)
        for layer in range(len(self.NN)):
            for node in range(len(self.NN[layer])):
                for weight in range(len(self.NN[layer][node])):
                    if self.pastError == None:
                        self.NN[layer][node][weight] += change[layer][node][weight]
                    else:
                        self.NN[layer][node][weight] += change[layer][node][weight] + (mf * self.pastError[layer][node][weight])
        self.pastError = change

    def deltaW(self, learn_rate, deltas, node_values, trainpoints):
        change = []
        for layer in reversed(range(len(self.NN))):
            layernode_change = []
            if layer != 0:
                for node in range(len(self.NN[layer])):
                    node_change = []
                    for inputnode in range(len(node_values[layer-1])):
                        weight_change = learn_rate * deltas[layer][node] * node_values[layer-1][inputnode]
                        node_change.append(weight_change)
                    node_change.append(learn_rate * deltas[layer][node])    #for bias node, since input value would be a 1
                    layernode_change.append(node_change)
                change.insert(0, layernode_change)
            if layer == 0:
                for node in range(len(self.NN[layer])):
                    node_change = []
                    for inputnode in range(len(trainpoints[:-1])):
                        weight_change = learn_rate * deltas[layer][node] * trainpoints[inputnode]
                        node_change.append((weight_change))
                    node_change.append(learn_rate * deltas[layer][node])  # for bias node, since input value would be a 1
                    layernode_change.append(node_change)
                change.insert(0, layernode_change)
        return(change)


    def softmax(self, new_node_vals):
        exp = np.exp(new_node_vals)
        prob = []
        for outputs in exp:
            prob.append(outputs/sum(exp))
        return(prob)

    def activation(self, dot):
        if self.classification == "classification":
            neg = np.negative(dot).astype('float128')
            val = 1/(1 + np.exp(neg))
            return val
        if self.classification == 'regression':
            return dot

    def backerror(self, output, expected):
        tot_error = 0
        errorarr = []
        for layer in reversed(range(len(output))):
            layererror = []
            if output[-1] == output[layer]: #if output layer, calculate error of output nodes
                nodeerror = []
                for node in range(len(output[layer])):
                    if self.classification == 'classification':
                        outputindex = output[layer].index(output[layer][node])
                        inputindex = self.classes.index(expected)
                        if outputindex == inputindex:
                            expected_val = 1
                        if outputindex != inputindex:
                            expected_val = 0
                    if self.classification == 'regression':
                        expected_val = float(expected)
                    error = expected_val - output[layer][node]        #CHANGING HERE
                    tot_error += error
                    nodeerror.append(error)
            else:   #if hidden layer use delta rule to calculate delta or error of hidden nodes
                for node in range(len(output[layer])):
                    error = 0
                    for errornode in range(len(errorarr[layer - 1])):
                        for weight in range(len(self.NN[layer][node])):
                            error += self.NN[layer][node][weight] * errorarr[layer - 1][errornode]  #summation of weight between node and node of next layer multiplied by error of next node
                        nodeerror.append(error)

            for node in range(len(output[layer])):
                newerror = nodeerror[node] * self.derivative(output[layer][node])
                layererror.append(newerror)
            errorarr.insert(0, layererror)
        return(errorarr)

    def derivative(self, output):
        if self.classification == 'classification':
            return output * (1 - output)
        if self.classification == 'regression':
            return 1





