import time
from network import Io

__author__ = 'Azatris'

import trainer
import network
import mnist_loader

import logging
import sys

""" Sandbox. """

# The logging initialization should be more general and taken out of run.py.
log = logging.root
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler_stream = logging.StreamHandler(sys.stdout)
handler_stream.setFormatter(formatter)
log.addHandler(handler_stream)

tr_d, va_d, te_d = mnist_loader.load_data_revamped()
t = trainer.Trainer()
stopper = trainer.Stopper(4)
architecture = [784, 400, 400, 10]
net = network.Network(architecture, 0.1)
t.sgd(
    net,
    tr_d,
    30,
    10,
    0.1,
    evaluation_data=te_d,
    monitor_evaluation_cost=False,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=False,
    monitor_training_accuracy=False,
    stopper=stopper
)
Io.save(
    net,
    "networks\\" + time.strftime("%Y%m%d-%H%M%S") + architecture + ".json"
)