__author__ = 'Azatris'

import numpy as np


class Utils(object):
    @staticmethod
    def vectorize_digit(i):
        """ Return a 10-dimensional one-hot vector with a 1.0 in the ith
        position and zeroes elsewhere.  This is used to convert a digit into a
        corresponding desired output from the neural network."""

        vectorized_digit = np.zeros((10,))
        vectorized_digit[i] = 1.0
        return vectorized_digit

    @staticmethod
    def vectorize_digits(digits):
        """ Return a len(digits)x10-dimensional one-hot vector with a 1.0 in
        the ith position and zeroes elsewhere.  This is used to convert a digit
        into a corresponding desired output from the neural network."""

        vectorized_digits = np.zeros((len(digits), 10))
        for idx, digit in enumerate(digits):
            vectorized_digits[idx][digits] = 1.0
        return vectorized_digits

    @staticmethod
    def shuffle_in_unison(feats, labels):
        rng_state = np.random.get_state()
        np.random.shuffle(feats)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        return feats, labels

    @staticmethod
    def softmax(v):
        exp_v = np.exp(v)
        return (exp_v.T / np.sum(exp_v, axis=1)).T + 1e-8


class CrossEntropyCost:
    def __init__(self):
        pass

    @staticmethod
    def fn(a, y):
        return np.mean(np.max(np.nan_to_num(-y*np.log(a)), axis=1))

    @staticmethod
    def delta(a, y):
        return a - y