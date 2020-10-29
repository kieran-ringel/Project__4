from Layer import Layer
class NN:
    def __init__(self, input, hlayers, hnodes, output):
        self.input = input
        self.hlayers = hlayers
        self.hnodes = hnodes
        self.output = output

    def getNN(self, input, hlayers, hnodes, output):
        NN = []
        if hlayers == 0:
            first_layer = Layer.getLayer(self, output, input)
        else:
            first_layer = Layer.getLayer(self, hnodes, input)
        NN.append(first_layer)
        for layer in range(hlayers - 1):
            hlayer = Layer.getLayer(self, hnodes, len(NN[-1]))
            NN.append(hlayer)
        if hlayers != 0:
            outputL = Layer.getLayer(self, output, len(NN[-1]))
            NN.append(outputL)
        return(NN)






