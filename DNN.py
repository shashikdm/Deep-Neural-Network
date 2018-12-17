#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 07:19:27 2018

@author: shashi
"""
import numpy as np
import matplotlib.pyplot as plt
#Class for trainer
class DNN:
    def __init__(self, layers, neurons, X_train, y_train, learning_rate = 0.01, iterations = 5000, activation_hidden = "relu", activation_output = "sigmoid", show_cost = True, plot_cost = True):
        self.layers = layers  #No. of layers = output layer + hidden layers
        self.X_train = np.array(X_train,ndmin = 2) #Input layer
        self.y_train = np.array(y_train,ndmin = 2) #output layer
        self.neurons = np.array(neurons) #List containing number of neurons in each layer
        self.neurons = np.append(self.X_train.shape[0], self.neurons)#0th layer is input
        self.weights = [] #Weights for each layer Remember indices start from 0
        self.biases = [] #Biases for each layer
        for i in range(0, layers): #Random initialisation with correct dimensions
            self.weights.append(np.random.randn(self.neurons[i+1],self.neurons[i])*0.01)
            self.biases.append(np.zeros((self.neurons[i+1],1)))
        self.activation_hidden = activation_hidden #Activation function for hidden layer
        self.activation_output = activation_output #Activation function of output layer
        self.learning_rate = learning_rate #The learning rate
        self.iterations = iterations#No. of iterations
        self.activations = []#stores activations used for backprop
        self.linears = []#stores Z values used for backprop
        self.costs = []#Stores costs to plot
        self.show_cost = show_cost;
        self.plot_cost = plot_cost
    def train(self):#call this to train the model
        for i in range(0, self.iterations):#outer loop for forward and backward props
            self.forward_prop(self.X_train)
            if i%100 == 0 and (self.show_cost or self.plot_cost):#pring every 100th value
                self.costs.append(self.compute_cost(self.activations[self.layers], self.y_train))
                if self.show_cost:
                    print("cost @", i,":", self.costs[-1])
            self.backward_prop()
        if self.plot_cost:#Plot
            plt.plot(np.squeeze(self.costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (x100)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()
            
    def forward_prop(self, X):
        A = X
        self.activations = []
        self.linears = []
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
    def backward_prop(self):
        AL = self.activations[self.layers]
        A = self.activations[self.layers-1]
        Z = self.linears[self.layers-1]
        #starting with derivative of cross entropy cost function for binary classification
        if self.activation_output == "sigmoid":
            dLdA = -(np.divide(self.y_train,AL)-np.divide(1-self.y_train,1-AL))/self.y_train.shape[1] 
            dAdZ = self.dsigmoid(Z)
            dLdZ = dLdA*dAdZ
            dLdW = np.dot((dLdZ),A.T)/self.y_train.shape[1]  #A = dZdW
            dLdb = np.sum(dLdZ, axis = 1, keepdims = True)/self.y_train.shape[1] 
            self.weights[self.layers-1] = self.weights[self.layers-1] - self.learning_rate*dLdW
            self.biases[self.layers-1] = self.biases[self.layers-1] - self.learning_rate*dLdb
            dLdA = np.dot(self.weights[self.layers-1].T,dLdA*dAdZ)
        elif self.activation_output == "softmax":
            dLdZ = (AL - self.y_train)
            dLdW = np.dot(dLdZ,A.T)/self.y_train.shape[1] #A = dZdW
            dLdb = np.sum(dLdZ, axis = 1, keepdims = True)/self.y_train.shape[1]
            self.weights[self.layers-1] = self.weights[self.layers-1] - self.learning_rate*dLdW
            self.biases[self.layers-1] = self.biases[self.layers-1] - self.learning_rate*dLdb
            dLdA = np.dot(self.weights[self.layers-1].T,dLdZ)
        for i in reversed(range(0, self.layers-1)):
            Z = self.linears[i]
            A = self.activations[i]
            if i < self.layers-1 and self.activation_hidden == "relu":
                dAdZ = self.drelu(Z)
            elif i < self.layers-1 and self.activation_hidden == "sigmoid":
                dAdZ = self.dsigmoid(Z)
            elif i < self.layers-1 and self.activation_hidden == "softmax":
                dAdZ = self.dsoftmax(Z)
            dLdW = np.dot((dLdA*dAdZ),A.T) #A = dZdW
            dLdb = np.sum(dLdA*dAdZ, axis = 1, keepdims = True)
            self.weights[i] = self.weights[i] - self.learning_rate*dLdW
            self.biases[i] = self.biases[i] - self.learning_rate*dLdb
            dLdA = np.dot(self.weights[i].T,dLdA*dAdZ)
        
    def relu(self, z):#Helper function relu #tested ok
        return np.maximum(0,z)
    def drelu(self, z):#Helper function derivative of relu #tested ok
        z[z<=0] = 0
        z[z>0] = 1
        return z
    
    def sigmoid(self, z):#Helper function sigmoid #tested ok
        return 1/(1+np.exp(-z))
    def dsigmoid(self, z):#Helper function derivative of sigmoid #tested ok
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def softmax(self, z):#helper function softmax #doubtful
        z = z - np.max(z)
        return np.exp(z)/np.sum(np.exp(z), axis = 0, keepdims = True)

    def dsoftmax(self, z):#helper function forderivative of softmax #doubtful
        return self.softmax(z)*(1-self.softmax(z))
    
    def compute_cost(self, AL, y):#Helper function cross entropy cost
        #cross entropy loss function for binary classification
        return -np.sum(np.dot(y.T,np.log(AL)))/y.shape[1]
    def predict(self, X_test):#returns probs and predictions
        self.forward_prop(X_test)
        probabilities = np.array(self.activations[self.layers],ndmin = 2)
        predictions = np.zeros((self.neurons[-1],X_test.shape[1]))
        for i in range(0,X_test.shape[1]):
            predictions[np.argmax(probabilities[...,i]),i] = 1
        return probabilities, predictions
        