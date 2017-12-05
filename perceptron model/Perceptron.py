#!/usr/bin/python
# -*- coding: utf-8 -*-

class Perceptron():
    def __init__(self, learn_rate, input_num, activat_func, iteration):
        self.bias_term = 0.0
        self.iterate_num = iteration
        self.learn_rate = learn_rate
        self.activat_func = activat_func
        self.weights = [0.0 for i in range(input_num)]

    #y = f(w*x + b)
    def predict(self, vector):

        prod = map(lambda(x, w): x*w, zip(vector, self.weights))
        add = reduce(lambda a, b: a + b, prod, 0.0)
        output = self.activat_func(add + self.bias_term)

        return output

    def single_iterate(self, input_vec, labels):
        
        pairs = zip(input_vec, labels)
        for (vector, actural) in pairs:
            predicted = self.predict(vector)
            
            #update the bias term and weight
            #wi = wi + delta(wi)
            #b = b + delta(b)
            #delta(w1) = learning_rate*(actural label - predicted label)*xi
            #delta(b) = learning_rate*(actural label - predicted label)
            delta = actural - predicted
            self.bias_term += delta*self.learn_rate
            self.weights = map(
                lambda(x, w): w + self.learn_rate*delta*x,
                zip(vector, self.weights))
            
    def train(self, input_data, labels):

        for i in xrange(self.iterate_num):
            self.single_iterate(input_data, labels)
        
   
