__author__ = 'Azatris'

import time
from evaluator import Evaluator
from network import Io
import numpy as np


from trainer import Trainer
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

# Subset the data
data_size = 50000
tr_d = (np.asarray(tr_d[0][:data_size]), np.asarray(tr_d[1][:data_size]))
te_d = (np.asarray(te_d[0][:data_size]), np.asarray(te_d[1][:data_size]))

trainer = Trainer()
evaluator = Evaluator(tr_d, te_d, log_interval=data_size/50)
#scheduler = scheduler.DecayScheduler()
architecture = [784, 800, 10]
net = network.Network(architecture, 0.1)
trainer.sgd(
    net,
    tr_d,
    10,
    evaluator=evaluator,
    scheduler=None
)
Io.save(
    net,
    "networks\\" + time.strftime("%Y%m%d-%H%M%S") + str(architecture) + ".json"
)