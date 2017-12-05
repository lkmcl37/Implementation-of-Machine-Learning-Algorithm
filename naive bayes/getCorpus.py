#-*- coding: UTF-8 -*-
import os
import nltk
from Serialization import*
from random import shuffle
from nltk.corpus import stopwords

class Parser():

        def __init__(self):       
                self.stop_words = set(stopwords.words('english'))

        def parseData(self, data):
                
                token = data.split(" ")
                normalized = [i.lower() for i in token if i.lower() not in self.stop_words]
               
                return normalized

        #This function is for obtaining data from the dataset
        #and divide the dataset into three parts:
        #80% for training set, 10% for test set, 10% for dev set
        def loadCorpus(self):

                pos_path = "movie_reviews/pos"
                neg_path = "movie_reviews/neg"

                pos_files= os.listdir(pos_path)
                neg_files= os.listdir(neg_path)

                category_word = []
                #category_word is list in which each element is a tuple (label, review)

                for txt in pos_files:
                        f = open(pos_path+"/"+txt)
                        iter_f = iter(f)
                        review = ""
                        for line in iter_f:
                                review += line
                        line_word = self.parseData(review)
                        category_word.append(tuple([1, line_word]))
                        
                for txt in neg_files:
                        f = open(neg_path+"/"+txt)
                        iter_f = iter(f)
                        review = ""
                        for line in iter_f:
                                review += line
                        line_word = self.parseData(review)
                        category_word.append(tuple([0, line_word]))

                #shuffle the whole dataset
                shuffle(category_word)
                
                train_size = int(len(category_word) * 0.8)
                test_size = int(len(category_word) * 0.1)

                train_set = category_word[:train_size]
                test_set = category_word[train_size:train_size+test_size]
                dev_set = category_word[train_size+test_size:]

                #serialize the dataset
                serializeFile(train_set, 'train_set')
                serializeFile(test_set, 'test_set')

        #create dictionary from the training set
        def getDict(self, train_set):

                lexicon = []
                for doc in train_set:
                        lexicon += doc[1]

                return set(lexicon)

        #deserialize the dataset
        def getData(self):

                train_set = deSerialize('training_data/train_set')
                test_set = deSerialize('training_data/test_set')

                return train_set, test_set

