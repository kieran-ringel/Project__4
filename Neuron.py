import random
class Neuron:
    def __init__(self, prevNodes):
        self.prevNodes = prevNodes

    def getNeuron(self, prevNodes):
        weights = []
        print(range(prevNodes + 1))
        for weight in range(prevNodes + 1): #add one to include bias:
            weights.append(random.uniform(-.01, .01))
        return(weights)
