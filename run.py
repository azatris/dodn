__author__ = 'Azatris'

import trainer
import network
import mnist_loader

import logging
import sys

log = logging.root
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler_stream = logging.StreamHandler(sys.stdout)
handler_stream.setFormatter(formatter)
log.addHandler(handler_stream)

tr_d, va_d, te_d = mnist_loader.load_data_revamped()
t = trainer.Trainer()
net = network.Network([784, 200, 10], 0.1)
t.sgd(
    net,
    tr_d,
    400,
    10,
    0.1,
    evaluation_data=va_d,
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True,
)