__author__ = 'Azatris'

import logging
import sys
import numpy as np

from trainer import Trainer
import network
import mnist_loader


""" Sandbox. """

# The logging initialization should be more general and taken out of run.py.
log = logging.root
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", "%H:%M:%S")
handler_stream = logging.StreamHandler(sys.stdout)
handler_stream.setFormatter(formatter)
log.addHandler(handler_stream)

tr_d, va_d, te_d = mnist_loader.load_data_revamped()

# Subset the data
data_size = 50000
tr_d = (np.asarray(tr_d[0][:data_size]), np.asarray(tr_d[1][:data_size]))

trainer = Trainer()
architecture = [784, 200, 200, 200, 10]
net = network.Network(architecture, 0.1)
trainer.mac(
    net,
    tr_d,
    va_d
)