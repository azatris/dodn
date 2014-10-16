
from evaluator import Evaluator
from utils import CrossEntropyCost, Utils

__author__ = 'Azatris'

import numpy as np
import logging

log = logging.root


class Trainer(object):
    def __init__(self, cost=CrossEntropyCost):
        np.random.seed(42)
        self.cost = cost

    def sgd(self, network, training_data, epochs,
            minibatch_size, learning_rate,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            log_interval=500):

        evaluator = Evaluator(
            self.cost, training_data, evaluation_data,
            monitor_training_cost, monitor_training_accuracy,
            monitor_evaluation_cost, monitor_evaluation_accuracy
        )
        
        feats, labels = training_data
        log.info("Starting SGD training with...")
        log.info("Feats \t%s", feats.shape)
        log.info("Labels \t%s", labels.shape)

        for epoch in xrange(epochs):
            log.info("Epoch \t%d", epoch)

            feats, labels = Utils.shuffle_in_unison(feats, labels)
            feats_split = np.split(feats, len(feats)/minibatch_size)
            labels_split = np.split(labels, len(labels)/minibatch_size)

            training_cost = 0.0
            count = 1
            for mini_feats, mini_labels in zip(feats_split, labels_split):
                training_cost += self.update(
                    network, mini_feats, mini_labels, learning_rate
                )

                # TODO: This functionality should not be here
                if count % log_interval == 0:
                    log.info(
                        "Cost after %d minibatches is %f",
                        count,
                        training_cost/count
                    )
                count += 1

            evaluator.monitor(network)

    def update(self, network, xs, ys, learning_rate):
        # generate predictions given current params
        activations = network.feed_forward(xs, return_all=True)

        # here we compute the top level error and cost for minibatch
        cost_gradient = self.cost.delta(activations[-1], ys)
        scalar_cost = self.cost.fn(activations[-1], ys)

        # backpropagate the errors, compute deltas
        deltas = network.feed_backward(cost_gradient, activations)
        
        # now, given both passes, compute actual gradients w.r.t params
        nabla_b = np.empty(len(network.layers), dtype=object)
        nabla_w = np.empty(len(network.layers), dtype=object)
        
        for idx in xrange(0, len(network.layers)):
            # we compute the sum over minibatch, and
            # compensate in learning rate scaling it down by mini-batch size
            nabla_b[idx] = np.sum(deltas[idx], axis=0)
            nabla_w[idx] = np.dot(deltas[idx].T, activations[idx]).T
            
        learning_rate_scaled = learning_rate/len(xs)
        
        for idx, layer in enumerate(network.layers):
            layer.weights -= learning_rate_scaled * nabla_w[idx]
            layer.biases -= learning_rate_scaled * nabla_b[idx]
        
        return scalar_cost
