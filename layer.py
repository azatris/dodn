__author__ = 'Azatris'

from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def feed_forward(self, inputs):
        pass

    @abstractmethod
    def feed_backward(self, gradients):
        pass


class Linear(Layer):
    def __init__(self, neurons, inputs_per_neuron, weight_magnitude):
        self.weights = np.random.uniform(
            -weight_magnitude, weight_magnitude, (neurons, inputs_per_neuron)
        )
        self.biases = np.zeros((neurons, 1))

    def feed_forward(self, inputs):
        # print "weights shape {}".format(np.shape(self.weights))
        # print "inputs shape {}".format(np.array(inputs).shape)
        # print "biases shape {}".format(np.shape(self.biases))
        return np.rollaxis(np.dot(self.weights, inputs), 1) + self.biases

    def feed_backward(self, gradients):
        # print "weights.T shape {}".format(self.weights.T.shape)
        # print "gradients shape {}".format(gradients.shape)
        return np.rollaxis(np.dot(self.weights.T, gradients), 1)  # check this


class Sigmoid(Linear):
    def __init__(self, neurons, inputs_per_neuron, weight_magnitude):
        super(Sigmoid, self).__init__(
            neurons, inputs_per_neuron, weight_magnitude
        )

    def feed_forward(self, inputs):
        linear_activations = super(Sigmoid, self).feed_forward(inputs)
        return 1.0 / (1.0 + np.exp(-linear_activations))  # check this

    def feed_backward(self, gradients):
        sigmoid_primes = gradients*(1.0 - gradients)
        return super(Sigmoid, self).feed_backward(sigmoid_primes)