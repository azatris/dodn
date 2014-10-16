from utils import Utils

__author__ = 'Azatris'

import numpy as np
import logging

log = logging.root


class Evaluator(object):
    def __init__(self, cost_function, training_data, evaluation_data,
                 monitor_training_cost, monitor_training_accuracy,
                 monitor_evaluation_cost, monitor_evaluation_accuracy):

        self.cost_function = cost_function
        self.training_data = training_data
        self.evaluation_data = evaluation_data
        self.monitor_training_cost = monitor_training_cost
        self.monitor_training_accuracy = monitor_training_accuracy
        self.monitor_evaluation_cost = monitor_evaluation_cost
        self.monitor_evaluation_accuracy = monitor_evaluation_accuracy

    def monitor(self, network):
        log.info("Training complete")

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

    @staticmethod
    def total_cost(cost_type, data, network, convert=False):
        cost = 0.0
        feats, labels = data
        feats_split = np.split(feats, len(feats)/10)
        labels_split = np.split(labels, len(labels)/10)
        for mini_feats, mini_labels in zip(feats_split, labels_split):
            a = network.feed_forward(mini_feats)
            if convert:
                mini_labels = Utils.vectorize_digits(mini_labels)
            cost += np.sum(cost_type.fn(a, mini_labels), axis=0)/len(data)
        return cost

    @staticmethod
    def accuracy(data, network, convert=False):
        feats, labels = data
        if convert:
            labels = np.argmax(labels, axis=1)
        feats_split = np.split(feats, len(feats)/10)
        labels_split = np.split(labels, len(labels)/10)
        accurate_results = 0
        for mini_feats, mini_labels in zip(feats_split, labels_split):
            mini_label_estimates = np.argmax(
                network.feed_forward(mini_feats), axis=1
            )
            accurate_results += np.sum(
                np.equal(mini_label_estimates, mini_labels)
            )
        return accurate_results