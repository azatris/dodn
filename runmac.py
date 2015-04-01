import pickle
import time

__author__ = 'Azatris'

import logging
import sys
import numpy as np

from trainer import Mac
import network
import mnist_loader
from utils import Utils
from evaluator import Evaluator


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

trainer = Mac()
architecture = [784, 400, 400, 10]
evaluator = Evaluator(tr_d, va_d)
net = network.Network(architecture, 0.1)
trainer.mac(
    net,
    tr_d,
    va_d,
    evaluator=evaluator
)
best_net = net
val_err = Utils.error_fraction(
    evaluator.accuracy(va_d, best_net), len(va_d[0])
)*100
eva_err = Utils.error_fraction(
    evaluator.accuracy(te_d, best_net), len(te_d[0])
)*100
curr_time = time.strftime("%Y%m%d-%H%M%S")
network.Io.save(
    best_net,
    "networks/" +
    curr_time +
    "_" + str(architecture).replace(' ', '') +
    "_mac_valerr" + str(val_err) +
    "_evaerr" + str(eva_err) +
    ".json"
)
pickle.dump(
    evaluator.training_costs,
    open("networks/" + curr_time + "_training_costs.p", "wb")
)
pickle.dump(
    evaluator.validation_costs,
    open("networks/" + curr_time + "_validation_costs.p", "wb")
)
pickle.dump(
    evaluator.training_errors,
    open("networks/" + curr_time + "_training_errors.p", "wb")
)
pickle.dump(
    evaluator.validation_errors,
    open("networks/" + curr_time + "_validation_errors.p", "wb")
)
