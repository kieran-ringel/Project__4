from Organization import Org
from NeuralNet import NeuralNet
def main():
    """ Kieran Ringel
    For each data set three lines are run in main.
    """

    #print('Breast Cancer')
    #cancer = Org('Data/breast-cancer-wisconsin.data', [-1], -1, [-1])
    #df = cancer.open()
    # ##NN(file, number hidden layers, number hidden nodes per layer)
    #NeuralNet(df, 0, 3, 'classification')

    print('glass')
    glass = Org('Data/glass.data', [-1], -1, [-1])
    df = glass.open()
    NeuralNet(df, 2, 6, 'classification')

    #print('soybean')
    #soybean = Org('Data/soybean-small.data', [-1], -1, [-1])
    #df = soybean.open()
    #NeuralNet(df, 2, 13, 'classification')

    # print('abalone')
    # abalone = Org('Data/abalone.data', [-1], -1, [0])
    # df = abalone.open()
    # print(df)

    #print('machine')
    #machine = Org('Data/machine.data', [-1], -1, [-1])
    #df = machine.open()
    #NeuralNet(df, 2, 3, 'regression')
    #print(df)

    # print('forest')
    # forest = Org('Data/forestfires.data', [0], -1, [2,3])
    # df = forest.open()
    # print(df)

if __name__ == '__main__':
    main()
