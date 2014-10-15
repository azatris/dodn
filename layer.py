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
    def feed_backward(self, gradients, h):
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

    def feed_backward(self, eh, h):
        return np.dot(eh, self.weights.transpose())


class Sigmoid(Linear):
    def __init__(self, neurons, inputs_per_neuron, weight_magnitude):
        super(Sigmoid, self).__init__(
            neurons, inputs_per_neuron, weight_magnitude
        )

    def feed_forward(self, inputs):
        linear_activations = super(Sigmoid, self).feed_forward(inputs)
        return 1.0 / (1.0 + np.exp(-linear_activations))

    def feed_backward(self, eh, h):
        deltas = eh*h*(1.0 - h)
        eh_prev = super(Sigmoid, self).feed_backward(deltas, h)
        return deltas, eh_prev 

 
class Softmax(Linear):
    def __init__(self, neurons, inputs_per_neuron, weight_magnitude):
        super(Softmax, self).__init__(
            neurons, inputs_per_neuron, weight_magnitude
        )
    
    def feed_forward(self, inputs):
        linear_activations = super(Softmax, self).feed_forward(inputs)
        ostate = np.exp(linear_activations)
        return ostate/(np.sum(ostate, axis=1)+1e-8)

    def feed_backward(self, eh, h):
        # here we assume the gradient w.r.t cost function was already computed
        # and passed here and that the cost is cross entropy with 1-of-K coding
        # which kicks-off off-diagonal elements of cross-dependency of targets
        # (i.e. off-diagonal elements of the Jacobian) i.e. deltas=eh
        eh_prev = super(Softmax, self).feed_backward(eh, h)
        return eh, eh_prev
