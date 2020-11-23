import random
import numpy as np

from Feedforward import Feedforward as ff

class GeneticAlgorithm:
    def __init__(self, initialNN, train, epochs, errorarray):
        self.initialNN = initialNN
        self.errorarray = errorarray
        self.newlistNN = []
        return(GeneticAlgorithm.start(self, self.newlistNN, self.initialNN, train, epochs, errorarray))

    def start(self, newlistNN, initialNN, train, epochs, errorarray):
        maxfitness = 9999
        generations = 10
        for i in range(generations):
            newerrorarray = []
            while len(newlistNN) < len(initialNN):
                parents = GeneticAlgorithm.selection(self, self.initialNN, errorarray)
                child = GeneticAlgorithm.crossover(self, parents)
                newlistNN.append(GeneticAlgorithm.mutation(self, child))
            for pop in range(len(newlistNN)):
                GAerror = 0
                self.NN = newlistNN[pop]
                for i in range(epochs): #iterates for a number of epochs, generally kept low to prevent overfitting
                    train = train.sample(frac=1).reset_index(drop=True)
                    for row, trainpoints in train.iterrows():    #iterate through training data points
                       node_values = ff.feedforward(self, trainpoints)  #feeds forward to calculate all node values for NN
                       GAerror += self.calcerror(node_values[-1], trainpoints['class'])
                GAerror /= epochs * len(train)
                errorarray.append(GAerror)
                if GAerror <= maxfitness:
                    bestNN = self.NN
                    maxfitness = GAerror
            errorarray = newerrorarray
            initialNN = newlistNN
        return(bestNN)


    def selection(self, NN, errorarray):
        tournamentsize = 4
        parents = []
        for parent in range(2):
            tournament = random.sample(errorarray, tournamentsize)
            selection = min(tournament)
            parents.append(NN[errorarray.index(selection)])
        return(parents)

    def crossover(self, parents):
        newNN = []
        for layer in range(len(parents[0])):
            newLayer = []
            for node in range(len(parents[0][layer])):
                newNode = []
                for weight in range(len(parents[0][layer][node])):
                    choice = random.randint(0,1)
                    newNode.append(parents[choice][layer][node][weight])
                newLayer.append(newNode)
            newNN.append(newLayer)
        return(newNN)

    def mutation(self, child):
        numbermutations = 25
        weight = 5
        for mutate in range(numbermutations):
            random_layer = np.random.randint(0, len(child))
            random_node = np.random.randint(0, len(child[random_layer]))
            random_weight = np.random.randint(0, len(child[random_layer][random_node]))
            random_value = np.random.uniform(-1.0, 1.0, 1)
            child[random_layer][random_layer][random_weight] += random_value[0] * weight
        return(child)
