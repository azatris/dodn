import copy
import sys

from scipy.optimize import minimize

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

    def mac(self, network, training_data):
        """ Does method of auxiliary coordinates (MAC) training on a given
        network and training data. """

        def w_step():
            def w_step_function(w):
                quux = np.sum(
                    (zed - layer.feed_forward(zs[idx_layer], w))**2
                )
                log.debug("quux: %s", quux)
                return quux


            log.debug("Deep copying old network with shape \t%s", np.shape(network.layers))
            old_network = copy.deepcopy(network)
            log.debug("Old network copy has shape \t%s",  np.shape(network.layers))

            log.debug("Start enumerating through layers...")
            for idx_layer, layer in enumerate(old_network.layers):
                log.debug(
                    "At layer number %d with shape %s", idx_layer, layer.weights.shape
                )

                # This might be turned into a single "minimize", not sure yet.
                for idx_weight, weight in enumerate(layer.weights.T):
                    log.debug(
                        "At weight number %d with shape %s", idx_weight, weight.shape
                    )

                    log.debug("Start minimizing the weight...")
                    zed = zs[idx_layer+1].T[idx_weight]
                    res = minimize(w_step_function, weight)
                    # log.debug("Weight minimized. Result: %s", res)

                    log.debug("Updating network...")
                    network.layers[idx_layer].weights.T[idx_weight] = res.x
                    log.debug("Network updated. New weight[0]: %s", res.x[0])

        def z_step():
            # TODO: reshape flat zs instead of going by every sample
            def z_layer_step_function(layer_zed):
                log.debug("STEP FUNCTION count: %d", count[0])
                count[0] += 1

                #  not sure about the indexes here
                multiplier = quadratic_penalty

                if idx_layer_zs is 0:
                    activations = feats[0]
                else:
                    # log.debug("idx_layer_zs-1 %d", idx_layer_zs-1)
                    # log.debug("idx_layer_z %d", idx_layer_z)
                    # log.debug("old_zs[idx_layer_zs-1][idx_layer_z] %s", np.shape(old_zs[idx_layer_zs-1][idx_layer_z]))
                    # log.debug("old_zs[idx_layer_zs-1] %s", np.shape(old_zs[idx_layer_zs-1]))
                    activations = network.layers[idx_layer_zs-1].feed_forward(
                        old_zs[idx_layer_zs-1][idx_layer_z]
                    )
                    if idx_layer_zs is len(old_zs) - 1:
                        multiplier = 1

                # log.debug("Activation shape: %s", np.shape(activations))
                returnable = np.sum(
                    (multiplier*0.5*(layer_zed - activations))**2
                )
                # log.debug("Returnable @ z_step_f: %s", np.shape(returnable))
                return returnable

            log.debug("Trying to optimise zs.")
            log.debug("zs shape %s", np.shape(zs))
            old_zs = copy.deepcopy(zs)
            for idx_layer_zs, layer_zs in enumerate(old_zs):
                log.debug("ZS LAYER idx: %d, Shape: %s", idx_layer_zs, np.shape(layer_zs))
                for idx_layer_z, layer_z in enumerate(layer_zs):
                    count = [1]
                    log.debug("Z Idx: %d, Shape: %s", idx_layer_z, np.shape(layer_z))
                    res = minimize(z_layer_step_function, layer_z)
                    zs[idx_layer_zs][idx_layer_z] = res.x

        feats, labels = training_data
        log.info("Starting MAC training with...")
        log.info("Feats \t%s", feats.shape)
        log.info("Labels \t%s", labels.shape)

        # Ideally, this works for multiple data points...
        # What's the chance, really?
        log.debug("Initializing zs...")
        zs = network.feed_forward(feats, return_all=True)
        zs[-1] = labels
        log.debug("Zs initialized. Shape \t%s, first shape: %s", zs.shape, zs[0].shape)

        tolerance = 0.01  # nested error threshold
        quadratic_penalty = 1  # aka mu
        nested_error_change = sys.maxint
        while nested_error_change > tolerance:
            log.debug("Starting W-step...")
            w_step()
            log.debug("W-step complete.")

            log.debug("Starting Z-step...")
            z_step()
            log.debug("W-step complete.")

            quadratic_penalty *= 10
            # compute nested_error_change

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
