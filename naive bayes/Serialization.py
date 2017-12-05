#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle

#this module is for serializing and deserializing the data needed for the classifier
def deSerialize(filename):
    with open (filename, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist

def serializeFile(file, name):
    with open(name, 'wb') as fp:
        pickle.dump(file, fp)
