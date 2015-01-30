import copy
import sys

from scipy.optimize import minimize

from utils import CrossEntropyCost, Utils


__author__ = 'Azatris'

import numpy as np
import logging
from scheduler import ListScheduler

log = logging.root


def feed_single(param, weight):
    pass


class Trainer(object):
    """ Trains the class i.e. changes the weight and biases of
    a given network using training data and a particular cost
    function which to compute the errors with. """

    def __init__(self, cost=CrossEntropyCost):
        np.random.seed(42)  # for consistent results
        self.cost = cost

    def sgd(self, network, training_data, minibatch_size,
            momentum=0.5, evaluator=None, scheduler=None):
        """ Does stochastic gradient descent training on a given
        network and training data for a number of epochs (times). """

        def update(xs, ys, speeds):
            """ The core of sgd given features xs and their respective
            labels ys. """

            # Generate predictions given current params.
            activations = network.feed_forward(xs, return_all=True)

            # Compute the top level error and cost for minibatch.
            cost_gradient = self.cost.delta(activations[-1], ys)
            scalar_cost = self.cost.fn(activations[-1], ys)

            # Backpropagate the errors, compute deltas.
            deltas = network.feed_backward(cost_gradient, activations)

            # Given both passes, compute actual gradients w.r.t params.
            nabla_b = np.empty(len(network.layers), dtype=object)
            nabla_w = np.empty(len(network.layers), dtype=object)

            # Compute the sum over minibatch, and compensate in
            # learning rate scaling it down by mini-batch size.
            for idx in xrange(0, len(network.layers)):
                nabla_b[idx] = np.sum(deltas[idx], axis=0)
                nabla_w[idx] = np.dot(deltas[idx].T, activations[idx]).T
            learning_rate_scaled = learning_rate/len(xs)

            # Update momentum layers
            for idx, speed in enumerate(speeds):
                speed[0] = \
                    momentum*speed[0] - learning_rate_scaled*nabla_w[idx]
                speed[1] = \
                    momentum*speed[1] - learning_rate_scaled*nabla_b[idx]

            # Update weights and biases.
            for idx, (layer, speed) in \
                    enumerate(zip(network.layers, speeds)):
                layer.weights += speed[0]
                layer.biases += speed[1]

            return scalar_cost, speeds

        if scheduler is None:
            scheduler = ListScheduler()

        feats, labels = training_data
        log.info("Starting SGD training with...")
        log.info("Feats \t%s", feats.shape)
        log.info("Labels \t%s", labels.shape)

        learning_rate = scheduler.get_learning_rate()
        while learning_rate > 0:
            log.info("Epoch \t%d", scheduler.epoch)

            # Prepare training data
            feats, labels = Utils.shuffle_in_unison(feats, labels)
            feats_split = np.split(feats, len(feats)/minibatch_size)
            labels_split = np.split(labels, len(labels)/minibatch_size)

            # Initialize speed layers (which are updated via momentum)
            speed_layers = np.asarray([
                (np.zeros_like(L.weights), np.zeros_like(L.biases))
                for L in network.layers
            ])

            # Descend the gradient
            training_cost = 0.0
            for mini_feats, mini_labels in zip(feats_split, labels_split):
                scalar_cost, speed_layers = update(
                    mini_feats, mini_labels, speed_layers
                )
                training_cost += scalar_cost
                if evaluator is not None:
                    evaluator.log_training_cost(training_cost)

            # Network evaluation and learning rate scheduling
            accuracy = None
            if evaluator is not None:
                accuracy = evaluator.monitor(network)
            if scheduler is not None:
                scheduler.compute_next_learning_rate(accuracy, network)
                learning_rate = scheduler.get_learning_rate()

        if hasattr(scheduler, 'highest_accuracy_network'):
            network = scheduler.highest_accuracy_network
            log.info(
                "Learning stopped. Highest accuracy: %d",
                scheduler.highest_accuracy
            )

        # For plotting use
        if evaluator is not None:
            return evaluator.errors, evaluator.training_costs
