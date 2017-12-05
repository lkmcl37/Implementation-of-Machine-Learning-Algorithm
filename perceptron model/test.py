#!/usr/bin/python
# -*- coding: utf-8 -*-
from Perceptron import Perceptron

def f(x):
    return 1 if x > 0 else 0

def train_and_test():
    input_vec = [[1, 1], [0, 1], [1, 0], [0, 0]]
    labels = [1, 0, 0, 0]

    #learning rate = 0.1
    #input number = 2
    #activation function = f
    #number of iterations = 10
    p = Perceptron(0.1, 2, f, 10)
    p.train(input_vec, labels)

    #test the perceptron
    print '1 and 1 = %d' % p.predict([1, 1])
    print '0 and 0 = %d' % p.predict([0, 0])
    print '1 and 0 = %d' % p.predict([1, 0])
    print '0 and 1 = %d' % p.predict([0, 1])

train_and_test()
