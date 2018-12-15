#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 07:19:27 2018

@author: shashi
"""
import numpy as np
#Class for trainer
class DNN:
    def __init__(self):
        self.layers = 1  #No. of layers = output layer + hidden layers
        self.neurons = [] #List containing number of neurons in each layer
        self.weights = [] #Weights for each layer
        self.biases = [] #Biases for each layer
        self.X_train = [] #Input layer
        self.y_train = [] #output layer
        self.activation_hidden = "relu" #Activation function for hidden layer
        self.activation_output = "sigmoid" #Activation function of output layer
        self.learning_rate = 0.01 #The learning rate
        self.iterations = 1000
        self.costs = []
        AL = np.array([[ 0.8,  0.9,  0.4],[0.45, 0.4, 0.6]])
        y = np.array([[1, 0, 0],[0, 1, 1]])
        print(self.compute_cost(AL, y))
    #def train(self):

    #def forward_prop(self):
        
    #def backward_prop(self):
        
    def relu(self, z):#Helper function relu #tested ok
        return (z+np.absolute(z))/2
        
    def sigmoid(self, z):#Helper function sigmoid #tested ok
        return 1/(1+np.exp(z))
    
    def compute_cost(self, AL, y):#Helper function cross entropy cost
        #return -np.sum(np.sum(np.dot(y, np.log(AL).T) + np.dot(1-y, np.log(1-AL).T), axis = 1, keepdims = True), axis = 0, keepdims = True)/y.shape[1]
        return -np.sum(np.sum(np.dot(y.T,np.log(AL)), axis = 1), axis = 0)/y.shape[1]
dnn = DNN()