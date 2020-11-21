from Neuron import Neuron
class Layer:
    def __init__(self, nodes):
        self.nodes = nodes

    def getLayer(self, nodes, preNodes):
        '''Kieran Ringel
        A layer is a list of nodes'''
        layer = []
        for node in range(nodes):   #gets all nodes to add to the layer
            node = Neuron.getNeuron(self, preNodes)
            layer.append(node)
        return(layer)