from utils import Utils

__author__ = 'Azatris'

from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(object):
    """ Abstract Layer. Layer has weights and biases. """
    __metaclass__ = ABCMeta

    def __init__(self):
        """ Initializes the layer with neurons in the form of
        providing weights that connect to it and biases. """
        pass

    @abstractmethod
    def feed_forward(self, inputs):
        """ Computes the output of a layer when given the inputs
        from the previous layer. """
        pass

    @abstractmethod
    def feed_backward(self, error, activation):
        """ Computes the delta of this layer (and possibly the
        error of the previous layer) given the error of this
        layer. """
        pass


class Linear(Layer):
    """ Linear Layer i.e. no neuron activation function. """

    def __init__(
            self, neurons=None, inputs_per_neuron=None, weight_magnitude=0.1,
            weights=None, biases=None
    ):
        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            np.random.seed(42)  # for consistent results
            self.weights = np.random.uniform(
                -weight_magnitude, weight_magnitude,
                (inputs_per_neuron, neurons)
            )
            self.biases = np.zeros(neurons)

        super(Layer, self).__init__()

    def feed_forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def feed_backward(self, error, activation):
        return np.dot(error, self.weights.transpose())


class Sigmoid(Linear):
    """ Sigmoid Layer. Extends the Linear Layer by providing
    a sigmoidal activation function for each of the neurons. """

    def __init__(
            self, neurons=None, inputs_per_neuron=None, weight_magnitude=0.1,
            weights=None, biases=None
    ):
        super(Sigmoid, self).__init__(
            neurons, inputs_per_neuron, weight_magnitude, weights, biases
        )

    def feed_forward(self, inputs):
        linear_activations = super(Sigmoid, self).feed_forward(inputs)
        return 1.0 / (1.0 + np.exp(-linear_activations))

    def feed_backward(self, error, activation):
        delta = error*activation*(1.0 - activation)
        previous_error = super(Sigmoid, self).feed_backward(error, activation)
        return delta, previous_error


class Softmax(Linear):
    """ Softmax Layer. Extends the Linear Layer by providing
    a Softmax activation function for each of the neurons.
    Normally used as the last layer of a network. """

    def __init__(
            self, neurons=None, inputs_per_neuron=None, weight_magnitude=0.1,
            weights=None, biases=None
    ):
        super(Softmax, self).__init__(
            neurons, inputs_per_neuron, weight_magnitude, weights, biases
        )

    def feed_forward(self, inputs):
        linear_activation = super(Softmax, self).feed_forward(inputs)
        return Utils.softmax(linear_activation)

    def feed_backward(self, error, activation):
        """ Assumes the gradient w.r.t cost function was already
        computed and passed here and that the cost is cross-entropy
        with 1-of-K coding which kicks-off off-diagonal elements of
        cross-dependency of targets (i.e. off-diagonal elements of
        the Jacobian). """

        previous_error = super(Softmax, self).feed_backward(error, activation)
        return error, previous_error
