__author__ = 'Azatris'

import layer
import numpy as np
import logging

log = logging.root


class Network(object):
    def __init__(self, architecture, initial_weight_magnitude):
        self.layers = np.append(
            [
                layer.Sigmoid(
                    neurons, inputs_per_neuron, initial_weight_magnitude
                )
                for neurons, inputs_per_neuron
                in zip(architecture[1:], architecture[:-2])
            ],
            layer.Softmax(
                architecture[-1], architecture[-2], initial_weight_magnitude
            )
        )

        for L in xrange(0, len(self.layers)):
            log.info(
                "Created weight matrix in layer %d: %s",
                L, self.layers[L].weights.shape
            )
            log.info(
                "Created biases vector in layer %d: %s",
                L, self.layers[L].biases.shape
            )

    def feed_forward(self, x, return_all=False):
        activations = np.empty(len(self.layers) + 1, dtype=object)
        activations[0] = x
        for L in xrange(0, len(self.layers)):
            activations[L+1] = self.layers[L].feed_forward(activations[L])
        if return_all:
            return activations
        return activations[-1]

    def feed_backward(self, output_error, activations):
        assert len(activations) == (len(self.layers) + 1)
        deltas = np.empty(len(self.layers), dtype=object)
        error = output_error
        for L in xrange(len(self.layers) - 1, -1, -1):
            deltas[L], error = self.layers[L].feed_backward(
                error, activations[L+1]
            )
        assert(deltas[-1] == error)
        return deltas