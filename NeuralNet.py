import pandas as pd
import numpy as np
import math
from NN import NN
class NeuralNet():
    def __init__(self, file, hlayers, hnodes, classification):
        self.file = file
        self.hlayers = hlayers
        self.hnodes = hnodes
        self.classification = classification
        self.NN = self.initNN(file, hlayers, hnodes, classification)
        self.tenfold(file)

    def initNN(self, file, hlayers, hnodes, classification):
        input = file.shape[1] - 1
        if classification == 'regression':
            output = 1
        if classification == 'classification':
            self.classes = list(file['class'].unique())
            output = file['class'].nunique()
        neuralNet = NN.getNN(self, input, hlayers, hnodes, output)
        return(neuralNet)

    def tenfold(self, file):
        # 10 fold cross validation by getting every 10th data point of the sorted data
        fold = [None] * 10
        for cv in range(10): #CHANGE VAD TO 10
            to_test = file.iloc[cv::10]
            #to_test.reset_index(drop=True, inplace=True)
            fold[cv] = to_test

        for foldnum in range(10):  # get test and train datasets
            print("Run", foldnum+1)  #prints run number
            test = fold[foldnum]     #gets test set
            train_list = fold[:foldnum] + fold[foldnum+1:]    #gets training set, everything besided the test set
            train = pd.concat(train_list)           #concatanates the 2 parts of the test set
            train.reset_index(drop=True, inplace=True)  #resets index on both
            test.reset_index(drop=True, inplace=True)
            self.train(train)

    def train(self, train):
        for row, trainpoints in train.iterrows():    #iterate through training data points
           #print("\n")
           node_values = self.feedforward(trainpoints)
           error = self.backerror(node_values, trainpoints['class'])
           self.backpropagate(error, node_values)

    def test(self, test):
        pass

    def feedforward(self, trainpoint):
        outputarray = []
        for layer in range(self.hlayers + 1):  # iterate through layers
            if layer == 0:
                node_vals = list(trainpoint[:-1])
            else:
                node_vals = new_node_vals
            new_node_vals = []
            for node in range(len(self.NN[layer])):
                cur_node = np.dot(self.NN[layer][node][:-1], node_vals) + self.NN[layer][node][-1]
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


    def backpropagate(self, error, node_values):
        eta = .7        #learning rate
        self.deltaW(eta, error, node_values)

    def deltaW(self, learn_rate, error, node_values):
        for layer in reversed(range(len(self.NN))):
            for node in range(len(self.NN[layer])):
                for weight in range(len(self.NN[layer][node])):
                    learn_rate * 1

    def softmax(self, new_node_vals):
        exp = np.exp(new_node_vals)
        prob = []
        for outputs in exp:
            prob.append(outputs/sum(exp))
        return(prob)

    def activation(self, dot):
        if self.classification == "classification":
            return 1/(1 + np.exp(np.negative(dot)))
        if self.classification == 'regression':     #linear activation CHNAGE???
            return dot

    def backerror(self, output, expected):
        tot_error = 0
        errorarr = []
        for layer in reversed(range(len(output))):
            layererror = []
            if output[-1] == output[layer]:
                nodeerror = []
                for node in range(len(output[layer])):
                    outputindex = output[layer].index(output[layer][node])
                    inputindex = self.classes.index(expected)
                    if outputindex == inputindex:
                        rt = 1
                    if outputindex != inputindex:
                        rt = 0
                    error = rt - output[layer][node]
                    tot_error += error
                    nodeerror.append(error)
            else:
                print('here', layer)
                for node in range(len(output[layer])):
                    error = 0
                    for errornode in range(len(output[layer - 1])):
                        for weight in reversed(range(len(self.NN[layer][node]))):
                            error += self.NN[layer][node][weight] * errorarr[layer - 1][errornode]
                        nodeerror.append(error)

            for node in range(len(output[layer])):
                newerror = nodeerror[node] * self.derivative(output[layer][node])
                layererror.append(newerror)
            errorarr.append(layererror)
        return(errorarr)

    def derivative(self, output):
        if self.classification == 'classification':
            return output * (1 - output)




