import logging
import time

__author__ = 'Azatris'

import numpy as np

log = logging.root


class Utils(object):
    """ Static methods that are not justified enough
    to be in any of the classes that use them. """

    @staticmethod
    def vectorize_digit(i):
        """ Return a 10-dimensional one-hot vector with a 1.0 in the
        ith position and zeroes elsewhere.  This is used to convert a
        digit into a corresponding desired output from the neural
        network. """

        vectorized_digit = np.zeros((10,))
        vectorized_digit[i] = 1.0
        return vectorized_digit

    @staticmethod
    def vectorize_digits(digits):
        """ Return a len(digits)x10-dimensional one-hot vector with a
        1.0 in the ith position and zeroes elsewhere.  This is used to
        convert a digit into a corresponding desired output from the
        neural network. """

        vectorized_digits = np.zeros((len(digits), 10))
        for idx, digit in enumerate(digits):
            vectorized_digits[idx][digits] = 1.0
        return vectorized_digits

    @staticmethod
    def shuffle_in_unison(feats, labels):
        """ Shuffles feats and labels on the first axis such that the
        change in ordering is identical in both. Equal length arrays
        assumed. """

        rng_state = np.random.get_state()
        np.random.shuffle(feats)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        return feats, labels

    @staticmethod
    def shuffle_in_unison_with_aux(feats, labels, aux_layers):
        """ Shuffles feats and labels on the first axis such that the
        change in ordering is identical in both. Equal length arrays
        assumed. The same is performed on auxiliary coordinates
        on every layer. """

        rng_state = np.random.get_state()
        np.random.shuffle(feats)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        for layer in aux_layers:
            np.random.set_state(rng_state)
            np.random.shuffle(layer)
        return feats, labels, aux_layers

    @staticmethod
    def softmax(v):
        """ Classic implementation of the Softmax algorithm. """

        # log.debug("Using softmax with input: %s shape: %s", v, np.shape(v))
        exp_v = np.exp(v)
        # log.debug("Exp: %s shape: %s", exp_v, np.shape(exp_v))
        if len(np.shape(exp_v)) is 1:
            total = np.sum(exp_v)
        else:
            total = np.sum(exp_v, axis=1)
        # time.sleep(1)
        return (exp_v.T / total).T + 1e-8


class CrossEntropyCost:
    """ Functions related to cross entropy cost. """

    def __init__(self):
        pass

    @staticmethod
    def fn(a, y):
        """ Computes the scalar cost related to activation a and
        actual label y. """

        return np.mean(np.max(np.nan_to_num(-y*np.log(a)), axis=1))

    @staticmethod
    def delta(a, y):
        """ Computes the cost gradient related to activation a and
        actual label y. """

        return a - y