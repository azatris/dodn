
from evaluator import Evaluator
from utils import CrossEntropyCost, Utils

__author__ = 'Azatris'

import numpy as np
import logging
from scheduler import ListScheduler

log = logging.root


class Trainer(object):
    """ Trains the class i.e. changes the weight and biases of
    a given network using training data and a particular cost
    function which to compute the errors with. """

    def __init__(self, cost=CrossEntropyCost):
        np.random.seed(42)  # for consistent results
        self.cost = cost

    def sgd(self, network, training_data, minibatch_size,
            evaluator=None, scheduler=None):
        """ Does stochastic gradient descent training on a given
        network and training data for a number of epochs (times). """

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

            # Descend the gradient
            training_cost = 0.0
            for mini_feats, mini_labels in zip(feats_split, labels_split):
                training_cost += self.update(
                    network, mini_feats, mini_labels, learning_rate
                )
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

    def update(self, network, xs, ys, learning_rate):
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

        # Update weights and biases.
        for idx, layer in enumerate(network.layers):
            layer.weights -= learning_rate_scaled * nabla_w[idx]
            layer.biases -= learning_rate_scaled * nabla_b[idx]
        
        return scalar_cost