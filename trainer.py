
from evaluator import Evaluator
from utils import CrossEntropyCost

__author__ = 'Azatris'

import numpy as np


class Trainer(object):
    def __init__(self, cost=CrossEntropyCost):
        self.cost = cost

    def shuffle_with_copy(self, mat_a, mat_b):
        assert len(mat_a) == len(mat_b)
        shuffled_a = np.empty(mat_a.shape, dtype=mat_a.dtype)
        shuffled_b = np.empty(mat_b.shape, dtype=mat_b.dtype)
        permutation = np.random.permutation(len(mat_a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = mat_a[old_index]
            shuffled_b[new_index] = mat_b[old_index]
        return shuffled_a, shuffled_b

    def sgd(self, network, training_data, epochs,
            mini_batch_size, learning_rate,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        #fix the evaluator so it can fit whatever I did with training
        evaluator = Evaluator(
            self.cost, training_data, evaluation_data,
            monitor_training_cost, monitor_training_accuracy,
            monitor_evaluation_cost, monitor_evaluation_accuracy
        )
        
        feats, labels = training_data
        
        #print feats[0]
        #print labels[0]
        
        mini_batch_idx = np.arange(mini_batch_size, feats.shape[0], mini_batch_size)
        # if you prefer to operate on the list of minibatches, you can call here
        # np.split(feats, mini_batch_idx) and the same for the labels
        # and you will get what expected, with possibly smaller last batch which you can either
        # ignore, or save for the next epoch, or whatever what make sense, including 
        # update with smaller batch, but need to adjust learning rate for this mini-batch accordingly later on
        # BTW: in general, you want to make training data feed in different (iterative) way, otherwise you end
        # up rewriting everything here for TIMIT and other bigger datasets
        
        log_freq=500
        for epoch in xrange(epochs):
            print feats.shape
            print labels.shape
            feats, labels = self.shuffle_with_copy(feats, labels)
            tr_cost = 0.0
            p_idx, cnt = 0, 1
            for idx in mini_batch_idx:
                tr_cost += self.update(network, (feats[p_idx:idx,:], labels[p_idx:idx]), learning_rate)
                p_idx=idx
                if cnt%log_freq==0:
                    print "Cost after %d minibatches is %f"%(cnt, tr_cost/cnt)
                cnt+=1
                
            # fix evaluator, you do not want to do second pass of fprop through dataset (that's expensive!) to get the cost.
            # For training purposes accumulating while training is more than enough, the same with accuracy on train set really
            # it is just for statistics how the training progresses, you do not care about very precise number here, save time!
            #evaluator.monitor(epoch, network)

    def update(self, network, batch, learning_rate):
        #xs, ys = map(list, zip(*batch))
        
        xs, ys = batch
        
        #generate predictions given current params
        activations = network.feed_forward(xs, return_all=True)    
        #here we compute the top level error and cost for minibatch
        cost_grad = self.cost.delta(activations[-1], ys)
        scalar_cost = self.cost.fn(activations[-1], ys)
        #backpropagate the errors, compute deltas
        deltas = network.feed_backward(cost_grad, activations)
        
        #now, given both passes, compute actual gradients w.r.t params
        nabla_b = [None]*len(network.layers)
        nabla_w = [None]*len(network.layers)
        
        for idx in xrange(0, len(network.layers)):
            # we compute the sum over minibatch, and compensate in learning rate scaling it down by mini-batch size
            nabla_b[idx] = np.sum(deltas[idx], axis=0);
            nabla_w[idx] = np.dot(deltas[idx].transpose(), activations[idx]).transpose()
            
            #print "Deltas shape in %d layer is %s"%(idx, deltas[idx].shape)
            #print "Nablas w shape in %d layer is %s"%(idx, nabla_w[idx].shape)
            #print "Nablas b shape in %d layer is %s"%(idx, nabla_b[idx].shape)
            
        learning_rate_scaled = learning_rate/xs.shape[0] #mean over minibatch
        
        for idx in xrange(0, len(network.layers)):
            network.layers[idx].weights = network.layers[idx].weights - learning_rate_scaled*nabla_w[idx]
            network.layers[idx].biases = network.layers[idx].biases - learning_rate_scaled*nabla_b[idx]
        
        return scalar_cost
        
        #this below looks like a really awkward (and inefficient) way to update params to me
        #not sure what was the intention and why you did this way
        #for layer, layer_nabla_b, layer_nabla_w in iterable:
        #    layer.weights = np.array([
        #        w - (learning_rate/len(batch))*nw
        #        for w, nw
        #        in zip(layer.weights, average(layer_nabla_w, axis=0).T)
        #    ])
        #    layer.biases = np.array([
        #        b - (learning_rate/len(batch))*nb
        #        for b, nb
        #        in zip(layer.biases, average(layer_nabla_b, axis=0))
        #    ])

    @staticmethod
    def forward_propagate(network, xs):
        batch_activation = xs
        batch_activations = [xs]
        for layer in network.layers:
            batch_activation = layer.feed_forward(batch_activation)
            batch_activations.append(batch_activation)
        return batch_activations

    def backward_propagate(self, network, errors, activations):
        pass
