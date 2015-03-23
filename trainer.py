import copy
import sys
import gc
import evaluator as eva  # likely temporary, so it doesnt shadow sgd

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
        self.cost = cost
        np.random.seed(42)  # for consistent results


class Mac(Trainer):
    def __init__(self):
        super(Mac, self).__init__()

    def postprocessing_step(self, feats, labels, network):
        prev_scalar_cost = sys.maxint
        while True:
            activations = network.feed_forward(feats, return_all=True)
            scalar_cost = self.cost.fn(activations[-1], labels)
            log.info("scalar_cost %f", scalar_cost)
            if abs(prev_scalar_cost - scalar_cost) < 0.001:
                break
            minibatch_size = 64
            feats, labels = Utils.shuffle_in_unison(feats, labels)
            feats_split = np.array_split(feats, len(feats) / minibatch_size)
            labels_split = np.array_split(labels, len(labels) / minibatch_size)
            eta = 0.1 / minibatch_size
            for mini_feats, mini_labels in zip(feats_split, labels_split):
                mini_activations = network.feed_forward(
                    mini_feats, return_all=True)
                error = self.cost.delta(mini_activations[-1], mini_labels)
                network.layers[-1].biases -= eta * np.sum(error, axis=0)
                network.layers[-1].weights -= eta * np.dot(
                    mini_activations[-2].T, error)
            prev_scalar_cost = scalar_cost

    def mac(self, network, training_data, validation_data):
        """ Does method of auxiliary coordinates (MAC) training on a given
        network and training data. """

        def w_step():
            def w_top_jac(params_flat):
                w = params_flat.reshape(params.shape)

                fkminus1 = layer.feed_forward(aux[idx_layer], w)
                de_dak = fkminus1 - aux[idx_layer+1]

                return np.ndarray.flatten(np.append(
                    np.dot(aux[idx_layer].T, de_dak), [np.sum(de_dak, axis=0)]))

            def w_hidden_jac(params_flat):
                w = params_flat.reshape(params.shape)

                fkminus1 = layer.feed_forward(aux[idx_layer], w)
                de_dfk = (fkminus1-aux[idx_layer+1]) * (fkminus1*(1-fkminus1))

                return np.ndarray.flatten(np.append(
                    np.dot(aux[idx_layer].T, de_dfk), [np.sum(de_dfk, axis=0)]))

            def w_cost(params_flat):
                w = params_flat.reshape(params.shape)

                returnable = np.sum((
                    aux[idx_layer+1] - layer.feed_forward(aux[idx_layer], w)
                )**2)
                log.debug("Returnable w_step_f: %s", returnable)
                return returnable

            log.debug("Start enumerating through layers...")
            for idx_layer, layer in reversed(list(enumerate(network.layers))):
                log.debug("At layer number %d with shape %s",
                          idx_layer, layer.weights.shape)

                if idx_layer == len(network.layers) - 1:
                    jac = w_top_jac
                else:
                    jac = w_hidden_jac

                params = np.append(layer.weights, [layer.biases], axis=0)

                log.debug("Start minimizing W step function...")
                res = minimize(w_cost, params,
                               method='Newton-CG',
                               jac=jac,
                               options={'disp': True, 'xtol': 100})
                log.debug("W step function minimized.")

                optimised_params = res.x.reshape(params.shape)
                network.layers[idx_layer].weights = optimised_params[:-1]
                network.layers[idx_layer].biases = optimised_params[-1]

                log.debug("Updated network with optimized weights.")

        def a_step():
            def aux_top_jac(aux_flat):
                layer_aux = aux_flat.reshape(aux_shape)

                return np.ndarray.flatten(np.dot(
                    layer.feed_forward(layer_aux) - aux[idx_layer_aux+1],
                    layer.weights.T))

            def aux_hidden_jac(aux_flat):
                layer_aux = aux_flat.reshape(aux_shape)

                fk = layer.feed_forward(layer_aux)

                return np.ndarray.flatten(np.dot(
                    (fk-aux[idx_layer_aux+1]) * (fk*(1-fk)), layer.weights.T))

            def aux_cost(aux_flat):
                layer_aux = aux_flat.reshape(aux_shape)

                returnable = np.sum((
                    0.5 * (1 if idx_layer_aux is len(aux)-2 else mu) * (
                        aux[idx_layer_aux+1] - layer.feed_forward(layer_aux))
                )**2)
                log.debug("Returnable a_step_f: %s", returnable)
                return returnable

            for idx_layer_aux, layer_aux in enumerate(aux):
                if idx_layer_aux not in [0, len(aux) - 1]:
                    log.debug("At layer number %d with shape %s",
                              idx_layer_aux, layer_aux.shape)

                    aux_shape = np.shape(layer_aux)
                    layer = network.layers[idx_layer_aux]

                    if idx_layer_aux == len(aux) - 2:
                        jac = aux_top_jac
                    else:
                        jac = aux_hidden_jac

                    log.debug("Start minimizing A step cost function...")
                    res = minimize(aux_cost, layer_aux,
                                   method='Newton-CG',
                                   jac=jac,
                                   options={'disp': True, 'xtol': 1000})
                    log.debug("A step cost function minimized. ")

                    aux[idx_layer_aux] = res.x.reshape(aux_shape)
                    log.debug("Updated aux by optimized aux.")

        feats, labels = training_data
        feats, labels = Utils.shuffle_in_unison(feats, labels)

        log.info("Starting MAC training with...")
        log.info("Feats \t%s", feats.shape)
        log.info("Labels \t%s", labels.shape)

        log.info("Pretraining with SGD...")
        scheduler = ListScheduler(max_epochs=1)
        Sgd().sgd(network, training_data, scheduler=scheduler)

        evaluator = eva.Evaluator(training_data, validation_data)
        log.info("Training cost: \t%d",
                 evaluator.total_cost(self.cost, training_data, network))
        evaluator.monitor(network)

        aux = network.feed_forward(feats, return_all=True)
        aux[-1] = labels
        log.debug("aux initialized.")

        tolerance = 0.01  # nested error threshold
        mu = 1  # aka mu
        nested_error_change = sys.maxint
        step = 0

        while nested_error_change > tolerance:
            step += 1

            log.info("Starting W-step...")
            w_step()
            log.info("W-step complete.")

            # log.debug("Forcing garbage collection...")
            gc.collect()

            log.info("Starting A-step...")
            a_step()
            log.info("A-step complete.")

            # log.debug("Forcing garbage collection...")
            gc.collect()

            mu *= 10

            log.info("Quadratic penalty increased. New QP: %d", mu)

            log.info("Training cost: \t%d",
                     evaluator.total_cost(self.cost, training_data, network))
            evaluator.monitor(network)

            # TODO: compute nested_error_change
            nested_error_change -= sys.maxint/2

        log.info("Starting post-processing...")
        self.postprocessing_step(feats, labels, network)
        log.info("Post-processing done.")

        log.info("Training cost: \t%d",
                 evaluator.total_cost(self.cost, training_data, network))
        evaluator.monitor(network)


class Sgd(Trainer):
    def __init__(self):
        super(Sgd, self).__init__()

    def sgd(self, network, training_data, minibatch_size=10,
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
                nabla_w[idx] = np.dot(activations[idx].T, deltas[idx])
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
                    evaluator.log_training_costs(training_cost)

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
            return evaluator.validation_errors, evaluator.training_costs
