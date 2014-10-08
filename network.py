__author__ = 'Azatris'

import layer


class Network(object):
    def __init__(self, architecture, initial_weight_magnitude):
        self.layers = [
            layer.Sigmoid(neurons, inputs_per_neuron, initial_weight_magnitude)
            for neurons, inputs_per_neuron
            in zip(architecture[1:], architecture[:-1])
        ]

    def feed_forward(self, x):
        a = [x]
        for L in self.layers:
            a = L.feed_forward(a)
        return a
