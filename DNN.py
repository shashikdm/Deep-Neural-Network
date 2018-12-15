#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 07:19:27 2018

@author: shashi
"""
import numpy as np
#Class for trainer
class DNN:
    def __init__(self, layers, neurons, X_train, y_train, learning_rate = 0.01, iterations = 5000, activation_hidden = "relu", activation_output = "softmax"):
        self.layers = layers  #No. of layers = output layer + hidden layers
        self.X_train = np.array(X_train,ndmin = 2) #Input layer
        self.y_train = np.array(y_train,ndmin = 2) #output layer
        self.neurons = np.array(neurons) #List containing number of neurons in each layer
        self.neurons = np.append(self.X_train.shape[0], self.neurons)#0th layer is input
        self.weights = [] #Weights for each layer Remember indices start from 0
        self.biases = [] #Biases for each layer
        for i in range(0, layers): #Random initialisation with correct dimensions
            self.weights.append(np.random.rand(self.neurons[i+1],self.neurons[i]))
            self.biases.append(np.zeros((self.neurons[i+1],1)))
        self.weights = np.asarray(self.weights)
        self.biases = np.asarray(self.biases)
        self.activation_hidden = activation_hidden #Activation function for hidden layer
        self.activation_output = activation_output #Activation function of output layer
        self.learning_rate = learning_rate #The learning rate
        self.iterations = iterations
        self.costs = []
        self.activations = []
        self.linears = []
    def train(self):
        self.forward_prop(self.X_train)
        self.backward_prop()
        
    def forward_prop(self, X):
        A = X
        self.activations.append(A)
        for i in range(0, self.layers):
            Z = np.dot(self.weights[i], A) + self.biases[i]
            if i < self.layers-1 and self.activation_hidden == "relu":
                A = self.relu(Z)
            elif i < self.layers-1 and self.activation_hidden == "sigmoid":
                A = self.sigmoid(Z)
            elif i < self.layers-1 and self.activation_hidden == "softmax":
                A = self.softmax(Z)
            elif i == self.layers-1 and self.activation_output == "relu":
                A = self.relu(Z)
            elif i == self.layers-1 and self.activation_output == "sigmoid":
                A = self.sigmoid(Z)
            elif i == self.layers-1 and self.activation_output == "softmax":
                A = self.softmax(Z)
            self.linears.append(Z)
            self.activations.append(A)
        self.costs.append(self.compute_cost(A, self.y_train))
    def backward_prop(self):
        AL = self.activations[self.layers]
        dLdA = -np.divide(self.y_train,AL)/self.y_train.shape[1]
        for i in reversed(range(0, self.layers)):
            Z = self.linears[i]
            A = self.activations[i]
            if i < self.layers-1 and self.activation_hidden == "relu":
                dAdZ = self.drelu(Z)
            elif i < self.layers-1 and self.activation_hidden == "sigmoid":
                dAdZ = self.dsigmoid(Z)
            elif i < self.layers-1 and self.activation_hidden == "softmax":
                dAdZ = self.dsoftmax(Z)
            elif i == self.layers-1 and self.activation_output == "relu":
                dAdZ = self.drelu(Z)
            elif i == self.layers-1 and self.activation_output == "sigmoid":
                dAdZ = self.dsigmoid(Z)
            elif i == self.layers-1 and self.activation_output == "softmax":
                dAdZ = self.dsoftmax(Z)
            dLdW = np.dot((dLdA*dAdZ),A.T) #A = dZdW
            dLdb = np.sum(dLdA*dAdZ, axis = 1, keepdims = True)
            self.weights[i] = self.weights[i] - self.learning_rate*dLdW
            self.biases[i] = self.biases[i] - self.learning_rate*dLdb
            dLdA = np.dot(self.weights[i].T,dLdA*dAdZ)
        
    def relu(self, z):#Helper function relu #tested ok
        return (z+np.absolute(z))/2
    def drelu(self, z):#Helper function derivative of relu
        z = np.divide(z,np.absolute(z))
        return (z+np.absolute(z))/2
    
    def sigmoid(self, z):#Helper function sigmoid #tested ok
        return 1/(1+np.exp(z))
    def dsigmoid(self, z):#Helper function derivative of sigmoid
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def softmax(self, z):#helper function softmax
        return np.exp(z)/np.sum(np.sum(np.exp(z), axis = 1, keepdims = True), axis = 0)
    def dsoftmax(self, z):#helper function forderivative of softmax
        return self.softmax(z)*(1-self.softmax(z))
    
    def compute_cost(self, AL, y):#Helper function cross entropy cost
        #return -np.sum(np.sum(np.dot(y, np.log(AL).T) + np.dot(1-y, np.log(1-AL).T), axis = 1, keepdims = True), axis = 0, keepdims = True)/y.shape[1]
        return -np.sum(np.sum(np.dot(y.T,np.log(AL)), axis = 1), axis = 0)/y.shape[1]
    def predict(self, X_test):
        self.forward_prop(X_test)
        guesses = self.activations(self.layers-1)
        return guesses
        