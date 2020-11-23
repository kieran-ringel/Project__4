import pandas as pd
import math

from NN import NN
from Feedforward import Feedforward as ff
from Backpropagate import Backpropagate as bp
from GeneticAlgorithm import GeneticAlgorithm as ga
from DifferentialEvolution import DifferentialEvoluation as de
class NeuralNet:
    def __init__(self, file, hlayers, hnodes, classification, type, population):
        print('hidden layers', hlayers)
        print('hidden nodes', hnodes)
        self.file = file
        self.hlayers = hlayers
        self.hnodes = hnodes
        self.type = type
        self.population = population
        self.pastError = None       #initalizes past error to None so that for the first data point momentum
                                    # cannot be calculated
        self.classification = classification
        self.temp = []
        for pop in range(population):
            self.temp.append(self.initNN(file, hlayers, hnodes, classification))
        print('what we want', self.temp[0])
        self.tenfold(file)


    def initNN(self, file, hlayers, hnodes, classification):
        '''Kieran Ringel
        Gets information needs to get shape of NN and then calls a function to get the shape'''
        input = file.shape[1] - 1
        if classification == 'regression':  #for regression there is one output node
            output = 1
        if classification == 'classification':  #for classification there is an output node for each class
            self.classes = list(file['class'].unique()) #creates a class variable to be reference later
            output = file['class'].nunique()
        neuralNet = NN.getNN(self, input, hlayers, hnodes, output) #initialized the shape of the NN
        return(neuralNet)

    def tenfold(self, file):
        '''Kieran Ringel
        Sets up 10 fold cross validation, each fold has every tenth data point'''
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
            self.train(train, foldnum)
            avgerror += self.test(test)
        print('Average Error', avgerror/10)

    def train(self, train, foldnum):
        '''Kieran Ringel
        Calls needed methods to feedforward and then back propagate the error'''
        #print(self.initialNN1)
        epochs = 1     #TUNE
        self.initialNN = self.temp
        errorarray = []
        for pop in range(self.population):
            GAerror = 0
            self.NN = self.initialNN[pop]    #sets NN to initalized NN, so for cross validation each fold starts with
                                            # a randomly initalized NN
            for i in range(epochs): #iterates for a number of epochs, generally kept low to prevent overfitting
                train = train.sample(frac=1).reset_index(drop=True)
                for row, trainpoints in train.iterrows():    #iterate through training data points
                   node_values = ff.feedforward(self, trainpoints)  #feeds forward to calculate all node values for NN
                   if self.type == "BP":
                       error = bp.backerror(self, node_values, trainpoints['class'])    #backpropagates
                                                                                        # the error on the output nodes
                       bp.backpropagate(self, error, node_values, trainpoints)  #uses backpropagated
                                                                                # error to change weights on the NN
                   if self.type == "GA":
                       GAerror += self.calcerror(node_values[-1], trainpoints['class'])
            if self.type == "GA":
                GAerror /= epochs * len(train)
                errorarray.append(GAerror)
        if self.type == "GA":
            self.NN = ga.__init__(self, self.initialNN, train, epochs, errorarray)
        if self.type == "DE":
            de.__init__(self, self.initialNN, train, epochs)

    def test(self, test):
        '''Kieran Ringel
        Calls methods to feed forward through trained NN and then calculated the error on the testing set'''
        tot_error = 0
        for row, testpoints in test.iterrows():
            node_values = ff.feedforward(self, testpoints)  #feedsforward to calculated all nodes for NN
            tot_error += self.calcerror(node_values[-1], testpoints['class'])   #looks at output nodes to calculate error
        print('error', tot_error/len(test))
        return(tot_error/len(test))

    def calcerror(self, output, expected):
        '''Kieran Ringel
        Calculates the error for the testing set'''
        error = 0
        if self.classification == 'classification': #calculates cross entropy
            for node in range(len(output)): #goes through all output nodes
                outputindex = output.index(output[node])    #gets index of output node
                inputindex = self.classes.index(expected)   #gets index of expected output
                if outputindex == inputindex:       #if they are the same rt = 1
                    rt = 1
                if outputindex != inputindex:       #otherwise rt = 0
                    rt = 0
                error -= (rt * math.log10(output[node])) #the error is the negated log of the probability of the expected outcome
        if self.classification == 'regression': #calculates squared error
            error = ((float(output[0]) - float(expected)) ** 2) / 2
        return(error)






