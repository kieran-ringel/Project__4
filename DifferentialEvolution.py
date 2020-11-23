import numpy as np

from Feedforward import Feedforward as ff

class DifferentialEvoluation:
    def __init__(self, initialNN, train, epochs):
        print('initlaizing')
        self.initialNN = initialNN
        DifferentialEvoluation.minimumerror = 99999
        DifferentialEvoluation.bestNN = []
        DifferentialEvoluation.start(self, self.initialNN, train, epochs)
        print('returning')
        return(DifferentialEvoluation.bestNN)

    def start(self, initialNN, train, epochs):
        generations = 10
        for i in range(generations):
            for target in initialNN:
                trial = DifferentialEvoluation.mutation(self, target)
                cross = DifferentialEvoluation.crossover(self, target, trial)
                initialNN[initialNN.index(target)] = DifferentialEvoluation.selection(self, target, cross, train, epochs)

    def mutation(self, target):
        beta = 1.5  #TUNE [0,2]
        distinct = [None] * 3
        count = 0
        while count != 3:
           rand = np.random.randint(0, len(self.initialNN))
           if (self.initialNN[rand] != target) and (self.initialNN[rand] not in distinct):
               distinct[count] = self.initialNN[rand]
               count += 1
        trial = []
        for layer in range(len(target)):
            newLayer = []
            for node in range(len(target[layer])):
                newNode = []
                for weight in range(len(target[layer][node])):
                    newNode.append(distinct[0][layer][node][weight] + (beta * (distinct[1][layer][node][weight] - distinct[2][layer][node][weight])))
                newLayer.append(newNode)
            trial.append(newLayer)
        return(trial)

    def crossover(self, target, trial):
        pr = .1     #TUNE [0,1]
        newNN = []
        for layer in range(len(target)):
            newLayer = []
            for node in range(len(target[layer])):
                newNode = []
                for weight in range(len(target[layer][node])):
                    choice = np.random.randint(0, 1)
                    if choice <= pr:
                        newNode.append(target[layer][node][weight])
                    else:
                        newNode.append(trial[layer][node][weight])
                newLayer.append(newNode)
            newNN.append(newLayer)
        return(newNN)

    def selection(self, target, cross, train, epochs):
        error = 0
        tests = [target, cross]
        errors = []
        for test in tests:
            self.NN = test
            for i in range(epochs):  # iterates for a number of epochs, generally kept low to prevent overfitting
                train = train.sample(frac=1).reset_index(drop=True)
                for row, trainpoints in train.iterrows():  # iterate through training data points
                    node_values = ff.feedforward(self, trainpoints)  # feeds forward to calculate all node values for NN
                    error += self.calcerror(node_values[-1], trainpoints['class'])
            error /= epochs * len(train)
            errors.append(error)
        index = errors.index(min(errors))
        if min(errors) < DifferentialEvoluation.minimumerror:
            DifferentialEvoluation.bestNN = tests[index]
        return(tests[index])


