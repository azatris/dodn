from evaluator import Evaluator
from utils import CrossEntropyCost

__author__ = 'Azatris'

import numpy as np


class Trainer(object):
    def __init__(self, cost=CrossEntropyCost):
        self.cost = cost

    def sgd(self, network, training_data, epochs,
            mini_batch_size, learning_rate,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        evaluator = Evaluator(
            self.cost, training_data, evaluation_data,
            monitor_training_cost, monitor_training_accuracy,
            monitor_evaluation_cost, monitor_evaluation_accuracy
        )

        for epoch in xrange(epochs):
            np.random.shuffle(training_data)
            total_training_data = len(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, total_training_data, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update(network, mini_batch, learning_rate)

            evaluator.monitor(epoch, network)

    def update(self, network, batch, learning_rate):
        xs, ys = map(list, zip(*batch))
        activations = self.forward_propagate(network, xs)
        nabla_b, nabla_w = self.backward_propagate(network, ys, activations)
        iterable = zip(network.layers, nabla_b, nabla_w)
        for layer, layer_nabla_b, layer_nabla_w in iterable:
            sum_layer_nabla_w = np.array(layer_nabla_w).sum(axis=0).T
            sum_layer_nabla_b = np.array(layer_nabla_b).sum(axis=0)
            layer.weights = np.array([
                w - (learning_rate/len(batch))*nw
                for w, nw in zip(layer.weights, sum_layer_nabla_w)
            ])
            layer.biases = np.array([
                b - (learning_rate/len(batch))*nb
                for b, nb in zip(layer.biases, sum_layer_nabla_b)
            ])

    @staticmethod
    def forward_propagate(network, xs):
        batch_activation = xs
        batch_activations = [xs]
        for layer in network.layers:
            batch_activation = layer.feed_forward(batch_activation)
            batch_activations.append(batch_activation)
        return batch_activations

    def backward_propagate(self, network, yz, batch_activations):
        nabla_b = [np.zeros(layer.biases.shape) for layer in network.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in network.layers]

        batch_delta = self.cost.delta(batch_activations[-1], yz)
        nabla_b[-1] = batch_delta
        nabla_w[-1] = [
            np.dot(delta, activations.T)
            for delta, activations
            in zip(batch_delta, batch_activations[-1])
        ]

        for idx, layer in reversed(list(enumerate(network.layers))):
            batch_delta = layer.feed_backward(batch_delta)
            nabla_b[idx-2] = batch_delta
            nabla_w[idx-2] = [
                np.dot(delta, activations.T)
                for delta, activations
                in zip(batch_delta, batch_activations[idx-2])
            ]

        return nabla_b, nabla_w
