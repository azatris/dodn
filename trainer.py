
from evaluator import Evaluator
from utils import CrossEntropyCost, Utils

__author__ = 'Azatris'

import numpy as np


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

        # fix the evaluator so it can fit whatever I did with training
        evaluator = Evaluator(
            self.cost, training_data, evaluation_data,
            monitor_training_cost, monitor_training_accuracy,
            monitor_evaluation_cost, monitor_evaluation_accuracy
        )
        
        feats, labels = training_data
        
        # print feats[0]
        # print labels[0]
        
        minibatch_idx = np.arange(
            minibatch_size, feats.shape[0], minibatch_size
        )
        # if you prefer to operate on the list of minibatches, you can call
        # np.split(feats, mini_batch_idx) and the same for the labels
        # and you will get what expected, with possibly smaller last batch
        # which you can either ignore, or save for the next epoch, or whatever
        # what make sense, including update with smaller batch, but need to
        # adjust learning rate for this mini-batch accordingly later on
        # BTW: in general, you want to make training data feed in
        # different (iterative) way, otherwise you end up rewriting everything
        # here for TIMIT and other bigger datasets

        for epoch in xrange(epochs):
            print feats.shape
            print labels.shape

            feats, labels = Utils.shuffle_in_unison(feats, labels)
            feats_split = np.split(feats, len(feats)/minibatch_size)
            labels_split = np.split(labels, len(labels)/minibatch_size)

            training_cost = 0.0
            cnt = 1
            for mini_feats, mini_labels in zip(feats_split, labels_split):
                training_cost += self.update(
                    network, (mini_feats, mini_labels), learning_rate
                )

                # TODO: clear this out
                if cnt % log_interval == 0:
                    print "Cost after {} minibatches is {}".format(
                        cnt, training_cost/cnt
                    )
                cnt += 1
                
            # fix evaluator, you do not want to do second pass of fprop through
            # dataset (that's expensive!) to get the cost. For training
            # purposes accumulating while training is more than enough, the
            # same with accuracy on train set really it is just for statistics
            # how the training progresses, you do not care about very precise
            # number here, save time!
            # evaluator.monitor(epoch, network)

    def update(self, network, batch, learning_rate):
        #xs, ys = map(list, zip(*batch))
        
        xs, ys = batch
        
        #generate predictions given current params
        activations = network.feed_forward(xs, return_all=True)    
        #here we compute the top level error and cost for minibatch
        cost_gradient = self.cost.delta(activations[-1], ys)
        scalar_cost = self.cost.fn(activations[-1], ys)
        #backpropagate the errors, compute deltas
        deltas = network.feed_backward(cost_gradient, activations)
        
        #now, given both passes, compute actual gradients w.r.t params
        nabla_b = np.empty(len(network.layers), dtype=object)
        nabla_w = np.empty(len(network.layers), dtype=object)
        
        for idx in xrange(0, len(network.layers)):
            # we compute the sum over minibatch, and compensate in learning
            # rate scaling it down by mini-batch size
            nabla_b[idx] = np.sum(deltas[idx], axis=0)
            nabla_w[idx] = np.dot(deltas[idx].T, activations[idx]).T
            
            # print "Deltas shape in %d layer is %s"%(idx, deltas[idx].shape)
            # print "Nablas w shape in %d layer is %s"%(idx, nabla_w[idx].shape)
            # print "Nablas b shape in %d layer is %s"%(idx, nabla_b[idx].shape)
            
        learning_rate_scaled = learning_rate/len(xs)
        
        for idx, layer in enumerate(network.layers):
            layer.weights -= learning_rate_scaled * nabla_w[idx]
            layer.biases -= learning_rate_scaled * nabla_b[idx]
        
        return scalar_cost

