from Organization import Org
from NeuralNet import NeuralNet
def main():
    """ Kieran Ringel
    For each data set three lines are run in main.
    The first one creates an instance of Org with the arguments being the file name, where the header
    is located to be removed ([-1] if there is no header, the location of the class so they can all
    be moved to the last column, and the column location of any categorical features so that one hot
    encoding can be applied.
    The next line calls to open the file and returns the dataframe of the file
    The final line creates an instance of NeuralNet with the arguments being the dataframe, the number
    of hidden layers, the number of hidden nodes per layer, whether classification or regression are
    to be performed, how the weights are being trained (GA, DE, BP, PSO), and the population size (1 for algorithms that
    don't need a population).
    """

    print('Breast Cancer')
    print("generations: 5 \n"
            "pop = 25 \n"
           "ts = 4 \n"
         "weight = 4 \n"
         "mutations = 10")
    #print("generations: 40 \n"
    #      "beta = 1.7 \n"
     #     "pop = 25 \n"
      #    "pr = .7")
    cancer = Org('Data/breast-cancer-wisconsin.data', [-1], -1, [-1])
    df = cancer.open()
    # ##NN(file, number hidden layers, number hidden nodes per layer)
    NeuralNet(df, 0, 12, 'classification', 'DE', 20)

    #print('glass')
    #glass = Org('Data/glass.data', [-1], -1, [-1])
    #df = glass.open()
    #NeuralNet(df, 2, 6, 'classification', "DE", 5)

    #print('soybean')
    #soybean = Org('Data/soybean-small.data', [-1], -1, [-1])
    #df = soybean.open()
    #NeuralNet(df, 0, 17, 'classification', "DE", 30)

    #print('abalone')
    #print("generations: 20 \n"
    #        "pop = 25 \n"
     #       "ts = 4 \n"
      #      "weight = 4 \n"
       #     "mutations = 30")
    #abalone = Org('Data/abalone.data', [-1], -1, [0])
    #df = abalone.open()
    #NeuralNet(df, 2, 1, 'regression', 'GA', 25)

    print('machine')
    machine = Org('Data/machine.data', [-1], -1, [-1])
    df = machine.open()
    NeuralNet(df, 2, 3, 'regression')
    print(df)

    #print('forest')
    #print("generations: 20 \n"
    #        "pop = 25 \n"
    #       "ts = 4 \n"
     #     "weight = 4 \n"
    #     "mutations = 30")
    #forest = Org('Data/forestfires.data', [0], -1, [-1])
    #df = forest.open()
    #NeuralNet(df, 0, 3, 'regression', 'BP', 10)

if __name__ == '__main__':
    main()
