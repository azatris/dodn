from utils import Utils

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
    def feed_backward(self, delta, activation):
        pass


class Linear(Layer):
    def __init__(self, neurons, inputs_per_neuron, weight_magnitude):
        np.random.seed(42)
        self.weights = np.random.uniform(
            -weight_magnitude, weight_magnitude, (inputs_per_neuron, neurons)
        )
        self.biases = np.zeros(neurons)
        super(Layer, self).__init__()

    def feed_forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def feed_backward(self, error, activation):
        return np.dot(error, self.weights.transpose())


class Sigmoid(Linear):
    def __init__(self, neurons, inputs_per_neuron, weight_magnitude):
        super(Sigmoid, self).__init__(
            neurons, inputs_per_neuron, weight_magnitude
        )

    def feed_forward(self, inputs):
        linear_activations = super(Sigmoid, self).feed_forward(inputs)
        return 1.0 / (1.0 + np.exp(-linear_activations))

    def feed_backward(self, error, activation):
        error *= activation*(1.0 - activation)
        eh_prev = super(Sigmoid, self).feed_backward(error, activation)
        return error, eh_prev


class Softmax(Linear):
    def __init__(self, neurons, inputs_per_neuron, weight_magnitude):
        super(Softmax, self).__init__(
            neurons, inputs_per_neuron, weight_magnitude
        )

    def feed_forward(self, inputs):
        linear_activation = super(Softmax, self).feed_forward(inputs)
        return Utils.softmax(linear_activation)

    def feed_backward(self, error, activation):
        # here we assume the gradient w.r.t cost function was already computed
        # and passed here and that the cost is cross entropy with 1-of-K coding
        # which kicks-off off-diagonal elements of cross-dependency of targets
        # (i.e. off-diagonal elements of the Jacobian)
        previous_error = super(Softmax, self).feed_backward(error, activation)
        return error, previous_error
