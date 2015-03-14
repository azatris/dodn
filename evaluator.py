from utils import Utils, CrossEntropyCost

__author__ = 'Azatris'

import numpy as np
import logging

log = logging.root


class Evaluator(object):
    """ Evaluates and logs the cost and accuracy of a given network.
    """

    def __init__(self, training_data, validation_data,
                 monitor_validation_cost=True,
                 monitor_training_cost=True,
                 monitor_training_accuracy=True,
                 cost_function=CrossEntropyCost,
                 log_interval=1000):

        self.cost_function = cost_function
        self.training_data = training_data
        self.validation_data = validation_data
        self.monitor_training_cost = monitor_training_cost
        self.monitor_training_accuracy = monitor_training_accuracy
        self.monitor_validation_cost = monitor_validation_cost
        self.log_interval = log_interval
        self.minibatches_count = 0
        self.training_errors = []
        self.validation_errors = []
        self.training_costs = []
        self.validation_costs = []

    def monitor(self, network):
        """ According to evaluator settings, evaluates and logs the
        cost and accuracy of a given network. """

        log.info("Training complete")

        if self.monitor_training_accuracy:
            training_accuracy = self.accuracy(
                self.training_data, network, convert=True
            )
            log.info(
                "Training accuracy: \t%d / %d",
                training_accuracy, len(self.training_data[0])
            )
            self.training_errors.append(
                Utils.error_fraction(
                    training_accuracy, len(self.training_data[0])
                )
            )

        if self.monitor_validation_cost:
            validation_cost = self.total_cost(
                self.cost_function, self.validation_data, network, convert=True
            )
            log.info(
                "Validation cost: %f",
                validation_cost
            )
            self.validation_costs.append(
                validation_cost
            )

        validation_accuracy = self.accuracy(self.validation_data, network)
        log.info(
            "Validation accuracy: \t%d / %d",
            validation_accuracy, len(self.validation_data[0])
        )
        self.validation_errors.append(
            Utils.error_fraction(
                validation_accuracy, len(self.validation_data[0])
            )
        )

        print
        self.minibatches_count = 0
        return validation_accuracy

    def log_training_costs(self, training_cost):
        self.minibatches_count += 1
        if self.monitor_training_cost:
            if self.log_interval > 0 and \
                    self.minibatches_count % self.log_interval == 0:
                log.info(
                    "Training cost after %d minibatches is %f",
                    self.minibatches_count,
                    training_cost/self.minibatches_count
                )
            self.training_costs.append(
                training_cost/self.minibatches_count
            )

    @staticmethod
    def total_cost(cost_type, data, network, convert=False, chunk_size=5000):
        """ Calculates the cost of given data against the network.
        :param convert: labels digit -> one-hot """

        cost = 0.0
        feats, labels = data
        chunks = len(feats)/chunk_size
        feats_split = np.split(feats, chunks)
        labels_split = np.split(labels, chunks)
        for mini_feats, mini_labels in zip(feats_split, labels_split):
            a = network.feed_forward(mini_feats)
            if convert:
                mini_labels = Utils.vectorize_digits(mini_labels)
            cost += cost_type.fn(a, mini_labels)
        return cost/np.ceil(chunks)

    @staticmethod
    def accuracy(data, network, convert=False, chunk_size=5000):
        """ Calculates the accuracy of given data against the network.
        :param convert: labels one-hot -> digit """

        feats, labels = data
        if convert:
            labels = np.argmax(labels, axis=1)
        if len(feats) > chunk_size and len(labels) > chunk_size:
            feats_split = np.split(feats, len(feats)/chunk_size)
            labels_split = np.split(labels, len(labels)/chunk_size)
        else:
            feats_split = [feats]
            labels_split = [labels]
        accurate_results = 0
        for mini_feats, mini_labels in zip(feats_split, labels_split):
            mini_label_estimates = np.argmax(
                network.feed_forward(mini_feats), axis=1
            )
            accurate_results += np.sum(
                np.equal(mini_label_estimates, mini_labels)
            )
        return accurate_results