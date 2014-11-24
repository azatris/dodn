from utils import Utils, CrossEntropyCost

__author__ = 'Azatris'

import numpy as np
import logging

log = logging.root


class Evaluator(object):
    """ Evaluates and logs the cost and accuracy of a given network.
    """

    def __init__(self, training_data, evaluation_data,
                 monitor_evaluation_cost=False,
                 monitor_evaluation_accuracy=True,
                 monitor_training_cost=False,
                 monitor_training_accuracy=False,
                 cost_function=CrossEntropyCost,
                 log_interval=1000):

        self.cost_function = cost_function
        self.training_data = training_data
        self.evaluation_data = evaluation_data
        self.monitor_training_cost = monitor_training_cost
        self.monitor_training_accuracy = monitor_training_accuracy
        self.monitor_evaluation_cost = monitor_evaluation_cost
        self.monitor_evaluation_accuracy = monitor_evaluation_accuracy
        self.log_interval = log_interval
        self.minibatches_count = 1

    def monitor(self, network):
        """ According to evaluator settings, evaluates and logs the
        cost and accuracy of a given network. """

        log.info("Training complete")
        accuracy = None

        if self.monitor_training_cost:
            cost = self.total_cost(
                self.cost_function, self.training_data, network
            )
            log.info("Training cost: \t%f", cost)
        if self.monitor_training_accuracy:
            accuracy = self.accuracy(self.training_data, network, convert=True)
            log.info(
                "Training accuracy: \t%d / %d",
                accuracy, len(self.training_data[0])
            )
        if self.monitor_evaluation_cost:
            cost = self.total_cost(
                self.cost_function, self.evaluation_data, network, convert=True
            )
            log.info("Evaluation cost: \t%f", cost)
        if self.monitor_evaluation_accuracy:
            accuracy = self.accuracy(self.evaluation_data, network)
            log.info(
                "Evaluation accuracy: \t%d / %d",
                accuracy, len(self.evaluation_data[0])
            )

        print

        return accuracy

    def log_training_cost(self, training_cost):
        if self.log_interval > 0 and \
                self.minibatches_count % self.log_interval == 0:
            log.info(
                "Cost after %d minibatches is %f",
                self.minibatches_count,
                training_cost/self.minibatches_count
            )
        self.minibatches_count += 1

    @staticmethod
    def total_cost(cost_type, data, network, convert=False):
        """ Calculates the cost of given data against the network.
        :param convert: labels digit -> one-hot """

        chunk_size = 4096
        cost = 0.0
        feats, labels = data
        feats_split = np.split(feats, len(feats)/chunk_size)
        labels_split = np.split(labels, len(labels)/chunk_size)
        for mini_feats, mini_labels in zip(feats_split, labels_split):
            a = network.feed_forward(mini_feats)
            if convert:
                mini_labels = Utils.vectorize_digits(mini_labels)
            cost += np.sum(cost_type.fn(a, mini_labels), axis=0)
        return cost/len(feats)

    @staticmethod
    def accuracy(data, network, convert=False):
        """ Calculates the accuracy of given data against the network.
        :param convert: labels one-hot -> digit """

        chunk_size = 4096
        feats, labels = data
        if convert:
            labels = np.argmax(labels, axis=1)
        feats_split = np.split(feats, len(feats)/chunk_size)
        labels_split = np.split(labels, len(labels)/chunk_size)
        accurate_results = 0
        for mini_feats, mini_labels in zip(feats_split, labels_split):
            mini_label_estimates = np.argmax(
                network.feed_forward(mini_feats), axis=1
            )
            accurate_results += np.sum(
                np.equal(mini_label_estimates, mini_labels)
            )
        return accurate_results