from utils import Utils

__author__ = 'Azatris'

import numpy as np


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

    def monitor(self, epoch, network):

        print "Epoch %s training complete" % epoch

        if self.monitor_training_cost:
            cost = self.total_cost(
                self.cost_function, self.training_data, network
            )
            print "Cost on training data: {}".format(cost)
        if self.monitor_training_accuracy:
            accuracy = self.accuracy(self.training_data, network, convert=True)
            print "Accuracy on training data: {} / {}".format(
                accuracy, len(self.training_data)
            )
        if self.monitor_evaluation_cost:
            cost = self.total_cost(
                self.cost_function, self.evaluation_data, network, convert=True
            )
            print "Cost on evaluation data: {}".format(cost)
        if self.monitor_evaluation_accuracy:
            accuracy = self.accuracy(self.evaluation_data, network)
            print "Accuracy on evaluation data: {} / {}".format(
                accuracy, len(self.evaluation_data)
            )

        print

    @staticmethod
    def total_cost(cost_type, data, network, convert=False):
        cost = 0.0
        for x, y in data:
            a = network.feed_forward(x)
            if convert:
                y = Utils.vectorize_digit(y)
            cost += cost_type.fn(a, y)/len(data)
        return cost

    @staticmethod
    def accuracy(data, network, convert=False):
        if convert:
            results = [
                (np.argmax(network.feed_forward(x)), np.argmax(y))
                for (x, y) in data
            ]
        else:
            results = [
                (np.argmax(network.feed_forward(x)), y)
                for (x, y) in data
            ]
        return sum(int(x == y) for (x, y) in results)