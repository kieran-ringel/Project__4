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
        '''Kieran Ringel
        Iterates over number of generations calling methods to perform selection, crossover and mutation.
        This creates a new population that replaces the previous population. The loss function is then calculated for
        all NN in the new generation. The best NN and it's weight are saved, to see if the next generation will
        produce a fitter NN. The new population then replaces the old. Once all the generations have gone through, the
        best NN is returned.'''
        maxfitness = 999999      #initialized to a high number so it will be replaced
        generations = 30        #TUNE
        for i in range(generations):    #loops through to keep updating best NN every generation as well as the population
            newerrorarray = []
            while len(newlistNN) < len(initialNN):  #uses generational replacement
                parents = GeneticAlgorithm.selection(self, self.initialNN, errorarray)
                child = GeneticAlgorithm.crossover(self, parents)
                newlistNN.append(GeneticAlgorithm.mutation(self, child))
            for pop in range(len(newlistNN)):   #once new population has been generated, goes through to get errors
                GAerror = 0
                self.NN = newlistNN[pop]
                for i in range(epochs): #iterates for a number of epochs, generally kept low to prevent overfitting
                    train = train.sample(frac=1).reset_index(drop=True)
                    for row, trainpoints in train.iterrows():    #iterate through training data points
                       node_values = ff.feedforward(self, trainpoints)  #feeds forward to calculate all node values for NN
                       GAerror += self.calcerror(node_values[-1], trainpoints['class'])
                GAerror /= epochs * len(train)
                errorarray.append(GAerror)      #appends to error list
                if GAerror <= maxfitness:       #if the most fit NN
                    bestNN = self.NN            #save NN
                    maxfitness = GAerror        #save error to be compared to
            errorarray = newerrorarray  #replaces population errors
            initialNN = newlistNN       #replaces population
        return(bestNN)


    def selection(self, NN, errorarray):
        """Kieran Ringel
        Tournament selection is used. A tournament is made by selecting the tournament size number of members from
        the loss list. The minimum loss in that list is then returned. The NN correlating to that loss is then
        added to the list of parents. Here 2 parents are used because that is what was discussed in class and
        it is what we are biologically familiar with."""
        tournamentsize = 8  #TUNE
        parents = []
        for parent in range(2):     #gets 2 parents as we typically think of
            tournament = random.sample(errorarray, tournamentsize)  #selects random errors that correlated to NN to fill tournament
            selection = min(tournament)                                #gets smallest error
            parents.append(NN[errorarray.index(selection)]) #add the parent that correlated to that smallest error
        return(parents) #returns list of 2 parents

    def crossover(self, parents):
        """Kieran Ringel
        Uniform cross over is performed to create a child by going through all of the weights in this NN sturcture
        and "flipping a coin"/ selecting one of the 2 parents to provide each weight. A child is returned.
        """
        newNN = []
        for layer in range(len(parents[0])):
            newLayer = []
            for node in range(len(parents[0][layer])):
                newNode = []
                for weight in range(len(parents[0][layer][node])):  #goes through each weight in NN
                    choice = random.randint(0,1)         #randomly selects which parent to take that weight from
                    newNode.append(parents[choice][layer][node][weight])    #appends it to make a new NN
                newLayer.append(newNode)
            newNN.append(newLayer)
        return(newNN)   #returns results child NN

    def mutation(self, child):
        '''Kieran Ringel
        For each mutation a random weight is selected and a random number is selected from a uniform distribution,
        this number is then multiplied by the mutation weight and is added to a random weight within the NN.
        '''
        numbermutations = 10    #TUNE
        weight = 5          #TUNE weight of the mutations
        for mutate in range(numbermutations):
            random_layer = np.random.randint(0, len(child))
            random_node = np.random.randint(0, len(child[random_layer]))
            random_weight = np.random.randint(0, len(child[random_layer][random_node])) #randomly selects weight to mutate
            random_value = np.random.uniform(-1.0, 1.0, 1)  #randomly selects mutation values from uniform distribution
            child[random_layer][random_node][random_weight] += random_value[0] * weight    #adds mutation value times the weight to the randomly selected weight to mutate
        return(child)
