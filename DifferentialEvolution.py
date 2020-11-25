import numpy as np

from Feedforward import Feedforward as ff

class DifferentialEvoluation:
    def __init__(self, initialNN, train, epochs):
        self.initialNN = initialNN
        DifferentialEvoluation.minimumerror = 99999
        DifferentialEvoluation.bestNN = []
        DifferentialEvoluation.start(self, self.initialNN, train, epochs)
        return(DifferentialEvoluation.bestNN)

    def start(self, initialNN, train, epochs):
        '''Kieran Ringel
        For each generation, it goes through each NN in the population and then mutates it and performs crossover, if
        the resulting trial vector has better fitness it replaces the target vector, otherwise the target vector remains
        in the population.'''
        generations = 30
        for i in range(generations):    #for each generation updates the population
            for target in initialNN:    #goes through each NN in the population
                trial = DifferentialEvoluation.mutation(self, target)
                cross = DifferentialEvoluation.crossover(self, target, trial)
                initialNN[initialNN.index(target)] = DifferentialEvoluation.selection(self, target, cross, train, epochs)

    def mutation(self, target):
        '''Kieran Ringel
        Gets three random distinct NN from the population. The resulting trial vector is the first NN1 + beta(NN2-NN3)'''
        beta = 1.3  #TUNE [0,2]
        distinct = [None] * 3
        count = 0
        while count != 3:   #gets three random NN
           rand = np.random.randint(0, len(self.initialNN))
           if (self.initialNN[rand] != target) and (self.initialNN[rand] not in distinct):  #makes sure the chose NN are distinct from eachother and the target
               distinct[count] = self.initialNN[rand]
               count += 1
        trial = []
        for layer in range(len(target)):    #goes through NN
            newLayer = []
            for node in range(len(target[layer])):  #goes through NN
                newNode = []
                for weight in range(len(target[layer][node])):  #goes through
                    newNode.append(distinct[0][layer][node][weight] + (beta * (distinct[1][layer][node][weight] - distinct[2][layer][node][weight])))   #trial vector is x1 + beta(x2-x3)
                newLayer.append(newNode)
            trial.append(newLayer)
        return(trial)   #returns trial vector

    def crossover(self, target, trial):
        '''Kieran Ringel
        Binomial crossover is then performed, for each weight a random number is generated, if it is below or equals
        hyperparameter pr then the weight from the target vector is used, otherwise the weight from the trial
        vector is used.'''
        pr = .3     #TUNE [0,1]
        newNN = []
        for layer in range(len(target)):
            newLayer = []
            for node in range(len(target[layer])):
                newNode = []
                for weight in range(len(target[layer][node])):
                    choice = np.random.randint(0, 1)
                    if choice <= pr:    #if randomly selected value is less than or = to hyperparameter pr
                        newNode.append(target[layer][node][weight]) #weight from target is used
                    else:
                        newNode.append(trial[layer][node][weight]) #else weight from trial is used
                newLayer.append(newNode)
            newNN.append(newLayer)
        return(newNN)

    def selection(self, target, cross, train, epochs):
        '''Kieran Ringel
        The loss function is calculated as the fitness for the target NN and the resulting NN, if the target NN performs
        better it remains in the population, otherwise the calculated NN replaces it. Throughout all of the generations
        the NN with the best fitness is saved.'''
        error = 0
        tests = [target, cross] #makes list of initial target NN and NN made by the DE
        errors = []
        for test in tests:  #gets loss function for both
            self.NN = test
            for i in range(epochs):  # iterates for a number of epochs, generally kept low to prevent overfitting
                train = train.sample(frac=1).reset_index(drop=True)
                for row, trainpoints in train.iterrows():  # iterate through training data points
                    node_values = ff.feedforward(self, trainpoints)  # feeds forward to calculate all node values for NN
                    error += self.calcerror(node_values[-1], trainpoints['class'])
            error /= epochs * len(train)
            errors.append(error)
        index = errors.index(min(errors))
        if min(errors) < DifferentialEvoluation.minimumerror:   #if it is the best NN that has been seen
            DifferentialEvoluation.bestNN = tests[index]    #sets the best NN to this NN
        return(tests[index])    #returns better NN


