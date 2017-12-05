#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from getCorpus import*
import operator
from collections import defaultdict

class Classifer():

    def __init__(self):
        self.ps = Parser()
        self.train_set, self.test_set = self.ps.loadCorpus()
        #self.lexicon = ['great', 'long', 'poor']
        self.lexicon = self.ps.getDict(self.train_set)

        self.pos_p = 0.
        self.neg_p = 0.

        self.pos_dic = defaultdict(lambda:1.0)
        self.neg_dic = defaultdict(lambda:1.0)

        self.total_num = 0.0
        self.pos_num = 2.0
        self.neg_num = 2.0

        self.mutual_info = {}

    #training the naive bayes classifier
    def train(self):

        all_word = []
        
        for i, doc in enumerate(self.train_set):
            if doc[0] == 1:
                for word in doc[1]:
                    if word in self.lexicon:
                        self.pos_dic[word] += 1
                        self.pos_num += 1
            else:
                for word in doc[1]:
                    if word in self.lexicon:
                        self.neg_dic[word] += 1
                        self.neg_num += 1
            all_word += doc[1]

        self.total_num = self.pos_num + self.neg_num
        self.pos_p = float(self.pos_num / self.total_num)
        self.neg_p = float(self.neg_num / self.total_num)

        all_word = set(all_word)
        for word in all_word:
            self.pos_dic[word] = float(self.pos_dic[word] / self.pos_num)
            self.neg_dic[word] = float(self.neg_dic[word] / self.neg_num)

    #Classifier
    def classify(self, doc):

        pos_score = 0.
        neg_score = 0.
        for word in doc:
            pos_score += math.log(self.pos_dic[word], 2)
            neg_score += math.log(self.neg_dic[word], 2)
            
        pos_score += math.log(self.pos_p, 2)
        neg_score += math.log(self.neg_p, 2)

        return 1 if pos_score > neg_score else 0

    #test(), compute() and average_performance() are for
    #testing the performance of the classifier
    def test(self):

        accuracy = 0.
        #true positive
        TP = 0.
        #false negative
        FN = 0.
        #false positive
        FP = 0.
        #true negative
        TN = 0.

        num_pos = 0
        num_neg = 0
        
        for text in self.test_set:
            gold = text[0]
            res = self.classify(text[1])
            if gold == 1:
                num_pos += 1
            else:
                num_neg += 1
            if gold == res:
                accuracy += 1
                if gold == 1 and res == 1:
                    TP += 1
                else:
                    TN += 1
            elif gold == 1 and res == 0:
                FN += 1
            elif gold == 0 and res == 1:
                FP += 1

        #The evaluation for possitive class
        F1 = self.compute(accuracy, TP, FN, FP, TN)
        #The evaluation for negative class
        F2 = self.compute(accuracy, TN, TP, FN, TP)
        #Compute the average performance
        self.average_performance(F1, F2, TP, FN, FP, TN, TN, TP, FN, TP)

    def compute(self, accuracy, TP, FN, FP, TN):

        print("The accuracy is: ")
        accuracy = accuracy/len(self.test_set)
        print accuracy 

        print("The precision is: ")
        precision = TP/(TP + FP)
        print precision

        print("The recall rate is: ")
        recall = TP/(TP + FN)
        print recall

        print("The F1 score is: ")
        F1 = (2*TP)/((2*TP) + FP + FN)
        print F1

        return F1

    def average_performance(self, F1, F2, TP1, FN1, FP1, TN1, TP2, FN2, FP2, TN2):
        print("The Micro average F1: ")
        micro_pre = (TP1+TP2)/(TP1+TP2+FP1+FP2)
        micro_recal = (TP1+TP2)/(TP1+TP2+FN1+FN2)
        micro_f1 = (2*micro_pre*micro_recal)/(micro_pre + micro_recal)
        print micro_f1

        print("The Macro average F1: ") 
        macro_f1 = (F1+F2)/2
        print macro_f1

    #computing the mutual information of the features
    def mutualInformation(self):
        
        for word in self.lexicon:
            mutual_info = 0
            prob_word_1 = float(self.pos_dic[word] + self.neg_dic[word])/self.total_num
            prob_word_0 = (float(self.pos_num - self.pos_dic[word]) + float(self.neg_num - self.neg_dic[word]))/self.total_num
        
            prob_word_pos = float(self.pos_dic[word])/self.total_num
            prob_word_neg = float(self.neg_dic[word])/self.total_num
    
            if (prob_word_1 > 0):
                if (prob_word_pos > 0): 
                    mutual_info = mutual_info + (prob_word_pos)*math.log(prob_word_pos/(self.pos_p*prob_word_1),2)
                if (prob_word_neg > 0): 
                    mutual_info = mutual_info + (prob_word_neg)*math.log(prob_word_neg/(self.neg_p*prob_word_1),2)
            prob_word_pos = (self.pos_num - self.pos_dic[word])/self.total_num
            prob_word_neg = (self.neg_num - self.neg_dic[word])/self.total_num
    
            if (prob_word_0 > 0):
                if (prob_word_pos > 0): 
                    mutual_info = mutual_info + (prob_word_pos)*math.log(prob_word_pos/(self.pos_p*prob_word_0),2)
                if (prob_word_neg > 0):
                    mutual_info = mutual_info + (prob_word_neg)*math.log(prob_word_neg/(self.neg_p*prob_word_0),2)
            self.mutual_info[word] = mutual_info

        self.sorted_mutual_info = sorted(self.mutual_info.items(), key = operator.itemgetter(1), reverse = True)

        for i in range(0, 3):
            print str(self.sorted_mutual_info[i][0]) + " " + str(self.sorted_mutual_info[i][1])+  "\n"

sample = Classifer()
sample.train()
sample.test()
#sample.mutualInformation()
