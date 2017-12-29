# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 09:26:17 2017

@author: Scott
"""

import os
os.chdir("./Documents/Code/Michael Neilson - Neural Networks and Deep Learning")


import mnist_loader
#import gzip

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network1_scott
import network2_scott
import time

import imp
imp.reload(network1_scott)
imp.reload(network2_scott)


t0 = time.time()
net = network1_scott.Network([784, 30, 10])
# data, epochs, batch_size, eta
net.SGD(training_data, 10, 20, 3.0, test_data=test_data)
t1 = time.time()
print("Network 2: ", t1-t0)


t0 = time.time()
net = network2_scott.Network([784, 30, 10])
# data, epochs, batch_size, eta
net.SGD(training_data, 10, 20, 3.0, test_data=test_data)
t1 = time.time()
print("Network 2: ", t1-t0)



