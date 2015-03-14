__author__ = 'Azatris'

import time
from evaluator import Evaluator
from network import Io
import numpy as np


from trainer import Sgd
import network
import mnist_loader
import scheduler

import logging
import sys
import pickle

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
# tr_d = (np.asarray(tr_d[0][:data_size]), np.asarray(tr_d[1][:data_size]))
# te_d = (np.asarray(te_d[0][:data_size]), np.asarray(te_d[1][:data_size]))

# Hyperparameters
lenargs = len(sys.argv)
lr = float(sys.argv[1]) if lenargs > 1 else 0.1
mom = float(sys.argv[2]) if lenargs > 2 else 0.6
dec = float(sys.argv[3]) if lenargs > 3 else 0.01
decthr = float(sys.argv[4]) if lenargs > 4 else 3
stopthr = float(sys.argv[5]) if lenargs > 5 else 10
mb = float(sys.argv[6]) if lenargs > 6 else 10

trainer = Sgd()
evaluator = Evaluator(tr_d, va_d, log_interval=data_size/50)
scheduler = scheduler.DecayScheduler(
    init_learning_rate=lr,
    decay=dec,
    decay_threshold=decthr,
    stop_threshold=stopthr
)
architecture = [784, 400, 400, 10]
net = network.Network(architecture, lr)
errors, training_costs = trainer.sgd(
    net,
    tr_d,
    mb,
    momentum=mom,
    evaluator=evaluator,
    scheduler=scheduler
)
error = (1 - float(evaluator.accuracy(te_d, net))/len(te_d[0]))*100
curr_time = time.strftime("%Y%m%d-%H%M%S")
Io.save(
    net,
    "networks/" +
    curr_time +
    "_" + str(architecture).replace(' ', '') +
    "_err" + str(error) +
    "_lr" + str(lr) +
    "_mom" + str(mom) +
    "_dec" + str(dec) +
    "_decthr" + str(decthr) +
    "_stopthr" + str(stopthr) +
    "_mb" + str(mb) +
    ".json"
)
pickle.dump(
    training_costs,
    open("networks/" + curr_time + "_mb_training_costs.p", "wb")
)
pickle.dump(
    errors,
    open("networks/" + curr_time + "_errors.p", "wb")
)
