__author__ = 'Azatris'

import numpy as np


class CrossEntropyCost:
    def __init__(self):
        pass

    @staticmethod
    def fn(a, y):
        return np.nan_to_num(np.sum(-y*np.log(a) - (1 - y)*np.log(1 - a)))

    @staticmethod
    def delta(a, y):
        return a - y


class Trainer(object):
    def __init__(self, cost=CrossEntropyCost):
        self.cost = cost

    def sgd(self, network, training_data, epochs,
            mini_batch_size, learning_rate,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        for j in xrange(epochs):
            np.random.shuffle(training_data)
            total_training_data = len(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, total_training_data, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update(network, mini_batch, learning_rate)

            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, network)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, network, convert=True)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, len(training_data)
                )
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, network, convert=True)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, network)
                print "Accuracy on evaluation data: {} / {}".format(
                    accuracy, len(evaluation_data)
                )
            print

    def update(self, network, batch, learning_rate):
        xs, ys = map(list, zip(*batch))
        activations = self.forward_propagate(network, xs)
        nabla_b, nabla_w = self.backward_propagate(network, ys, activations)
        iterable = zip(network.layers, nabla_b, nabla_w)
        # print "iterable shape {}".format(np.shape(iterable))
        # print "layer.biases {}, nabla {}".format(np.shape(network.layers[1].biases), np.shape(nabla_b))
        for layer, layer_nabla_b, layer_nabla_w in iterable:
            # print "layer weight shape {}".format(layer.weights.shape)
            # print "layer nabla w shape {}".format(np.array(layer_nabla_w).shape)
            sum_layer_nabla_w = np.array(layer_nabla_w).sum(axis=0).T
            sum_layer_nabla_b = np.array(layer_nabla_b).sum(axis=0)
            # print "weights {}, sum_nabla {}".format(np.shape(layer.weights), np.shape(sum_layer_nabla_w))
            layer.weights = np.array([
                w - (learning_rate/len(batch))*nw
                for w, nw in zip(layer.weights, sum_layer_nabla_w)
            ])
            layer.biases = np.array([
                b - (learning_rate/len(batch))*nb
                for b, nb in zip(layer.biases, sum_layer_nabla_b)
            ])
        # print "layer.biases {}, nabla {}".format(np.shape(network.layers[1].biases), np.shape(np.array(layer_nabla_b).sum(axis=0)))


    @staticmethod
    def forward_propagate(network, xs):
        batch_activation = xs
        # print "batch activation shape {}".format(np.shape(batch_activation))
        batch_activations = [xs]  # layer by layer
        for layer in network.layers:
            # not sure if this broadcasts over all xs
            batch_activation = layer.feed_forward(batch_activation)
            batch_activations.append(batch_activation)
        return batch_activations

    def backward_propagate(self, network, yz, batch_activations):
        # print "weights shape {}".format(np.shape(self.weights))
        # print "biases shape {}".format(np.shape(self.biases))
        nabla_b = [np.zeros(layer.biases.shape) for layer in network.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in network.layers]

        # last layer
        batch_delta = self.cost.delta(batch_activations[-1], yz)
        # print "1delta shape {}".format(np.array(batch_delta).shape)
        # print "1activation.T shape {}".format(batch_activations[-1].T.shape)
        # print batch_delta[0]
        nabla_b[-1] = batch_delta
        nabla_w[-1] = [
            np.dot(delta, activations.T)
            for delta, activations
            in zip(batch_delta, batch_activations[-1])
        ]

        # subsequent layers
        for idx, layer in reversed(list(enumerate(network.layers))):
            batch_delta = layer.feed_backward(batch_delta)
            # print "delta shape {}".format(np.array(batch_delta).shape)
            # print "activation.T shape {}".format(batch_activations[idx-2].T.shape)
            nabla_b[idx-2] = batch_delta  # check for out of bounds errors
            nabla_w[idx-2] = [
                np.dot(delta, activations.T)
                for delta, activations
                in zip(batch_delta, batch_activations[idx-2])
            ]

        return nabla_b, nabla_w

    def accuracy(self, data, network, convert=False):
        if convert:
            results = [
                (np.argmax(self.feed_forward(network, x)), np.argmax(y))
                for (x, y) in data
            ]
        else:
            results = [
                (np.argmax(self.feed_forward(network, x)), y)
                for (x, y) in data
            ]
        return sum(int(x == y) for (x, y) in results)

    @staticmethod
    def feed_forward(network, x):
        a = [x]
        for layer in network.layers:
            a = layer.feed_forward(a)
        return a

    @staticmethod
    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    def total_cost(self, data, network, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(network, x)
            if convert:
                y = self.vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        return cost


