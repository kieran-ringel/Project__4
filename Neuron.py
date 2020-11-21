import random
class Neuron:
    def __init__(self, prevNodes):
        self.prevNodes = prevNodes

    def getNeuron(self, prevNodes):
        '''Kieran Ringel
        Using the number of nodes from the previous layer, the weights on
        each node is initalized'''
        weights = []
        for weight in range(prevNodes + 1): #add one to include bias:
            weights.append(random.uniform(-.01, .01))   #the weights are initalized between -.01 and .01
        return(weights)
