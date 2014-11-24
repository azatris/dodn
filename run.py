import time
from evaluator import Evaluator
from network import Io

__author__ = 'Azatris'

import trainer
import network
import mnist_loader
import scheduler

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
evaluator = Evaluator(
    tr_d, te_d
)
scheduler = scheduler.DecayScheduler()
architecture = [784, 800, 10]
net = network.Network(architecture, 0.1)
t.sgd(
    net,
    tr_d,
    30,
    10,
    0.1,
    evaluator=evaluator,
    scheduler=scheduler
)
Io.save(
    net,
    "networks\\" + time.strftime("%Y%m%d-%H%M%S") + str(architecture) + ".json"
)