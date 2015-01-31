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
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler_stream = logging.StreamHandler(sys.stdout)
handler_stream.setFormatter(formatter)
log.addHandler(handler_stream)

tr_d, va_d, te_d = mnist_loader.load_data_revamped()

# Subset the data
data_size = 10
tr_d = (np.asarray(tr_d[0][:data_size]), np.asarray(tr_d[1][:data_size]))

trainer = Trainer()
architecture = [784, 20, 10]
net = network.Network(architecture, 0.1)
trainer.mac(
    net,
    tr_d
)