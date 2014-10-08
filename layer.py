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
        super(Layer, self).__init__()

    def feed_forward(self, inputs):
        return np.dot(self.weights, inputs) + self.biases

    def feed_backward(self, gradients):
        return np.dot(self.weights.T, gradients)


class Sigmoid(Linear):
    def __init__(self, neurons, inputs_per_neuron, weight_magnitude):
        super(Sigmoid, self).__init__(
            neurons, inputs_per_neuron, weight_magnitude
        )

    def feed_forward(self, inputs):
        linear_activations = super(Sigmoid, self).feed_forward(inputs)
        return 1.0 / (1.0 + np.exp(-linear_activations))

    def feed_backward(self, gradients):
        sigmoid_primes = gradients*(1.0 - gradients)
        return super(Sigmoid, self).feed_backward(sigmoid_primes)