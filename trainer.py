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

    def mac(self, network, training_data, validation_data):
        """ Does method of auxiliary coordinates (MAC) training on a given
        network and training data. """

        def w_step():
            def w_top_jac(w_flat):
                w = w_flat.reshape(ws_shape)

                # jacobian = np.dot(zs[idx_layer].T, zs[idx_layer+1] - labels)

                activations = network.layers[idx_layer].feed_forward(
                    zs[idx_layer], w
                )

                # log.debug(
                #     "w_top_jac zs[idx_layer] "
                #     "%s activations %s, "
                #     "zs[idx_layer+1] %s",
                #     zs[idx_layer].shape,
                #     activations.shape,
                #     zs[idx_layer+1].shape
                # )

                difference = activations - zs[idx_layer+1]

                jacobian = np.append(
                    np.dot(
                        zs[idx_layer].T,
                        difference
                    ),
                    [np.sum(difference, axis=0)]
                )

                return np.ndarray.flatten(jacobian)

            def w_hidden_jac(w_flat):
                w = w_flat.reshape(ws_shape)

                activations = layer.feed_forward(zs[idx_layer], w)

                de_dzk = activations - zs[idx_layer+1]

                # log.debug(
                #     "w_hidden_jac activations %s ",
                #     activations.shape
                # )jacobian

                dzk_dfk = activations*(1 - activations)

                # log.debug(
                #     "w_hidden_jac de_dzk %s dzk_dfk %s, zs[idx_layer] %s",
                #     de_dzk.shape, dzk_dfk.shape, zs[idx_layer].shape
                # )

                de_dzk_dzk_dfk = de_dzk*dzk_dfk

                jacobian = np.append(
                    np.dot(
                        zs[idx_layer].T,
                        de_dzk_dzk_dfk,
                    ),
                    [np.sum(de_dzk_dzk_dfk, axis=0)]
                )

                return np.ndarray.flatten(jacobian)

            def w_step_function(w_flat):
                w = w_flat.reshape(ws_shape)

                # log.debug("W step function initiated.")

                # log.debug(
                #     "Feeding forward %s using weights %s",
                #     zs[idx_layer].shape, w.shape
                # )

                activations = layer.feed_forward(zs[idx_layer], w)
                # log.debug("activations %s", activations.shape)

                difference = zed - activations
                # log.debug("difference %s", difference.shape)

                square = difference**2
                # log.debug("square %s", square.shape)

                value = np.sum(square)
                log.debug("W step function value: %s", value)

                # time.sleep(0.1)
                return value

            # log.debug(
            #     "Deep copying old network with shape \t%s",
            #     np.shape(network.layers)
            # )
            old_network = copy.deepcopy(network)
            # log.debug(
            #     "Old network copy has shape \t%s",
            #     np.shape(network.layers)
            # )

            log.debug("Start enumerating through layers...")
            for idx_layer, layer in \
                    reversed(list(enumerate(old_network.layers))):
                log.debug(
                    "At layer number %d with shape %s",
                    idx_layer, layer.weights.shape
                )

                zed = zs[idx_layer+1]

                if idx_layer == len(old_network.layers) - 1:
                    jac = w_top_jac
                else:
                    jac = w_hidden_jac

                params = np.append(layer.weights, [layer.biases], axis=0)
                ws_shape = params.shape

                log.debug("Start minimizing W step function...")
                res = minimize(
                    w_step_function,
                    params,
                    method='Newton-CG',
                    jac=jac,
                    options={
                        'disp': True,
                        'xtol': 100
                    }
                )
                log.debug("W step function minimized.")

                # log.debug("res.x %s", np.shape(res.x))
                optimised_params = res.x.reshape(ws_shape)

                network.layers[idx_layer].weights = optimised_params[:-1]
                network.layers[idx_layer].biases = optimised_params[-1]

                log.debug("Updated network with optimized weights.")

        def z_step():
            def z_top_jac(flat_layer_zeds):
                layer_zeds = flat_layer_zeds.reshape(zs_shape)

                weights = network.layers[idx_layer_zs].weights

                activations = \
                    network.layers[idx_layer_zs].feed_forward(layer_zeds)

                # log.debug("z_top_jac weights %s labels %s, oldzs %s",
                #           weights.shape,
                #           labels.shape,
                #           old_zs[idx_layer_zs].shape
                # )

                jacobian = np.dot(
                    activations - old_zs[idx_layer_zs+1],
                    weights.T
                )

                # log.debug("z_top_jac jacobian %s", jacobian.shape)

                return np.ndarray.flatten(jacobian)

            def z_hidden_jac(flat_layer_zeds):
                layer_zeds = flat_layer_zeds.reshape(zs_shape)

                # log.debug(
                #     "BLA1: %s",
                #     network.layers[idx_layer_zs-1].weights.shape
                # )
                # log.debug("BLA2: %s", old_zs[idx_layer_zs-1].shape)

                activations = network.layers[idx_layer_zs].feed_forward(
                    layer_zeds
                )

                # log.debug(
                #     "z_hidden_jac activation %s ",
                #     activations.shape
                # )

                de_dzkplus1 = activations - old_zs[idx_layer_zs+1]

                dzkplus1_dfkplus1 = activations*(1 - activations)

                weights = network.layers[idx_layer_zs].weights

                # log.debug(
                #     "z_hidden_jac de_dzkplus1 %s "
                #     "dzkplus1_dfkplus1 %s "
                #     "weights %s",
                #     de_dzkplus1.shape, dzkplus1_dfkplus1.shape, weights.shape
                # )

                # if that work can just transpose the whole thing instead
                jacobian = np.dot(de_dzkplus1*dzkplus1_dfkplus1, weights.T)

                # log.debug("z_hidden_jac jacobian %s", jacobian.shape)

                return np.ndarray.flatten(jacobian)

            def z_layer_step_function(flat_layer_zeds):
                layer_zeds = flat_layer_zeds.reshape(zs_shape)

                log.debug("STEP FUNCTION count: %d", count[0])
                count[0] += 1

                multiplier = quadratic_penalty

                # log.debug("idx_layer_zs-1 %d", idx_layer_zs-1)
                # log.debug("idx_layer_z %d", idx_layer_zs)
                # log.debug("old_zs[idx_layer_zs-1] %s",
                #           np.shape(old_zs[idx_layer_zs-1]))
                # log.debug("layer_zeds %s", np.shape(layer_zeds))

                activations = network.layers[idx_layer_zs].feed_forward(
                    layer_zeds
                )
                if idx_layer_zs is len(old_zs) - 2:
                    multiplier = 1

                # log.debug("Activation shape: %s", np.shape(activations))

                returnable = np.sum(
                    (multiplier*0.5*(old_zs[idx_layer_zs+1] - activations))**2
                )
                log.debug("Returnable z_step_f: %s", returnable)

                return returnable

            # log.debug("Trying to optimise zs.")
            # log.debug("zs shape %s", np.shape(zs))
            old_zs = copy.deepcopy(zs)
            for idx_layer_zs, layer_zs in reversed(list(enumerate(old_zs))):

                # zs contains labels and inputs, but we don't optimise them
                if idx_layer_zs is not len(old_zs) - 1 \
                        and idx_layer_zs is not 0:

                    log.debug(
                        "At layer number %d with shape %s",
                        idx_layer_zs, layer_zs.shape
                    )

                    zs_shape = np.shape(layer_zs)

                    count = [0]

                    if idx_layer_zs == len(old_zs) - 2:
                        jac = z_top_jac
                    else:
                        jac = z_hidden_jac

                    log.debug("Start minimizing Z step function...")
                    res = minimize(
                        z_layer_step_function,
                        layer_zs,
                        method='Newton-CG',
                        jac=jac,
                        options={
                            'disp': True,
                            'xtol': 1000
                        }
                    )
                    log.debug("Z step function minimized. ")

                    zs[idx_layer_zs] = res.x.reshape(zs_shape)
                    log.debug("Updated Z by optimized Z.")

        feats, labels = training_data
        feats, labels = Utils.shuffle_in_unison(feats, labels)

        log.info("Starting MAC training with...")
        log.info("Feats \t%s", feats.shape)
        log.info("Labels \t%s", labels.shape)

        log.info("Pretraining with SGD...")
        scheduler = ListScheduler(max_epochs=1)
        Sgd().sgd(network, training_data, scheduler=scheduler)

        evaluator = eva.Evaluator(training_data, validation_data)
        log.info(
            "Training cost: \t%d", eva.Evaluator.total_cost(
                self.cost, training_data, network
            )
        )
        evaluator.monitor(network)


        zs = network.feed_forward(feats, return_all=True)
        zs[-1] = labels
        log.debug("Z initialized.")

        tolerance = 0.01  # nested error threshold
        quadratic_penalty = 1  # aka mu
        nested_error_change = sys.maxint
        step = 0

        while nested_error_change > tolerance:
            step += 1

            log.info("Starting W-step...")
            w_step()
            log.info("W-step complete.")

            # log.debug("Forcing garbage collection...")
            gc.collect()

            log.info("Starting Z-step...")
            z_step()
            log.info("Z-step complete.")

            # log.debug("Forcing garbage collection...")
            gc.collect()

            quadratic_penalty *= 5

            log.info(
                "Quadratic penalty increased. New QP: %d",
                quadratic_penalty
            )

            log.info(
                "Training cost: \t%d", eva.Evaluator.total_cost(
                    self.cost, training_data, network
                )
            )
            evaluator.monitor(network)

            # TODO: compute nested_error_change
            nested_error_change = 0

        log.info("Starting post-processing...")
        while True:
            activations = network.feed_forward(feats, return_all=True)
            scalar_cost = self.cost.fn(activations[-1], labels)
            error = self.cost.delta(activations[-1], labels)
            log.info("scalar_cost %f", scalar_cost)
            if scalar_cost < 0.001:
                break
            network.layers[-1].biases -= 0.1*np.sum(error, axis=0)
            network.layers[-1].weights -= 0.1*np.dot(activations[-2].T, error)
        log.info("Post-processing done.")

        log.info(
            "Training cost: \t%d", eva.Evaluator.total_cost(
                self.cost, training_data, network
            )
        )
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
