#Importeren van de nodige libraries
import random

# Third-party libraries
import numpy as np

class NeuralNetwork(object):
	def __init__(self,sizes): #the list sizes contains the number of neurons in the respective layers.
		self.num_layers = len(sizes)  #the number of the layers in Network
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) 
						for x,y in zip(sizes[:-1],sizes[1:])]

	def feedforward(self,a):
		"""Return the output of the network if "a" is input"""
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a) + b)		
		return a
		
	def GDA(self, training_data, epochs, mini_batch_size,eta,
	        test_data = None):
		if test_data: 
			n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)        #rearrange the training_data randomly
			mini_batches = [ training_data[k:k + mini_batch_size]
			                               for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			if test_data:
				print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test))
			else:
				print("Epoch {0} complete".format(j))
			
	def update_mini_batch(self,mini_batch,eta):
		
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]
	
	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
        # backward pass
		delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
		return (output_activations-y)
		
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

import sys
sys.path.append("../")

import Mnist_loader as mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = NeuralNetwork([784, 30,  6, 10])
net.GDA(training_data, 5, 10, 3.0, test_data=test_data)
