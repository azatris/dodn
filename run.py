__author__ = 'Azatris'

import trainer
import network
import mnist_loader

tr_d, va_d, te_d = mnist_loader.load_data_revamped()
t = trainer.Trainer()
net = network.Network([784, 200, 10], 0.1)
t.sgd(
    net,
    tr_d[:1000],
    400,
    10,
    0.1,
    evaluation_data=te_d,
    monitor_training_cost=True
)