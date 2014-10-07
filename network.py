__author__ = 'Azatris'

import layer


class Network(object):
    def __init__(self, architecture, initial_weight_magnitude):
        self.layers = [
            layer.Sigmoid(neurons, inputs_per_neuron, initial_weight_magnitude)
            for neurons, inputs_per_neuron
            in zip(architecture[1:], architecture[:-1])
        ]


