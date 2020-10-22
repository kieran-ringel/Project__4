from Neuron import Neuron
class Layer:
    def __init__(self, nodes):
        self.nodes = nodes

    def getLayer(self, nodes, preNodes):
        layer = []
        for node in range(nodes):
            node = Neuron.getNeuron(self, preNodes)
            layer.append(node)
        return(layer)