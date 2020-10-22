import pandas as pd
import numpy as np
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
            output = file['class'].nunique()
        neuralNet = NN.getNN(self, input, hlayers, hnodes, output)
        return(neuralNet)

    def tenfold(self, file):
        # 10 fold cross validation by getting every 10th data point of the sorted data
        fold = [None] * 10
        for cv in range(10):
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
            self.train(self.NN, train)

    def train(self, NN, train):
       eta = .7    #learning rate
       print(NN[0][1])
       if self.classification == "classification": #use sigmoidal activation function
           for row, trainpoints in train.iterrows():    #iterate through training data points
               self.feedforward(trainpoints)

    def feedforward(self, trainpoint):
        for layer in range(self.hlayers + 1):  # iterate through layers
            print('layer',layer)
            if layer == 0:
                node_vals = list(trainpoint[:-1])
            else:
                node_vals = new_node_vals
            new_node_vals = []
            for node in range(len(self.NN[layer])):
                cur_node = np.dot(self.NN[layer][node][:-1], node_vals) + self.NN[layer][node][-1]
                new_node_vals.append(cur_node)
            new_node_vals = self.activation(new_node_vals)   #send to method
            print(new_node_vals)
            print(NN[layer])


    def backpropagate(self):
        pass

    def activation(self, dot):  #change method
        if self.classification == "classification":
            return 1/(1 + np.exp(np.negative(dot)))
        if self.classification == 'regression':
            #linear
            return dot


    def test(self, test):
        pass



    def netj(self):
        pass
