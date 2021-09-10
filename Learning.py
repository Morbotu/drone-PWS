#Benoemen variabelen en opstellen functie van een neural network
import math as mh
class Network(object):

    def __init__(self, sizes):
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) 
                            for x, y in zip(sizes[:-1], sizes[1:])]
#De random functie genereert willekeurige w en b met een Gausskromme met standdaarddeviatie 1 en gemiddelde 0
def sigmoid(z):
    return 1.0/(1.0+mh.exp(-z))

