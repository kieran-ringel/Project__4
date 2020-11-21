from Layer import Layer
class NN:
    def __init__(self, input, hlayers, hnodes, output):
        self.input = input
        self.hlayers = hlayers
        self.hnodes = hnodes
        self.output = output

    def getNN(self, input, hlayers, hnodes, output):
        """Kieran Ringel
        The NN is made up of layers that contain nodes that contain weights
        for this reason the NN just has layers for the hidden layers and output layer
        to know how many weights each node has, it must know the size of the previous layer"""
        NN = []
        if hlayers == 0:      #if there are no hidden layers, the only layer is the output layer, with its previous layer being the input
            first_layer = Layer.getLayer(self, output, input)
        else:   #otherwise the first layer is the first hidden layer
            first_layer = Layer.getLayer(self, hnodes, input)
        NN.append(first_layer)
        for layer in range(hlayers - 1):    #adds on remaining hidden layers
            hlayer = Layer.getLayer(self, hnodes, len(NN[-1]))
            NN.append(hlayer)
        if hlayers != 0:    #if the output layer wasn't added as the only layer, add the output layer
            outputL = Layer.getLayer(self, output, len(NN[-1]))
            NN.append(outputL)
        return(NN)






