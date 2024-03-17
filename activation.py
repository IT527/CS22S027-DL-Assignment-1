import numpy as np

"""#Activation Functions"""

def sigmoid(z): #Sigmoid Activation Function
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z): #Softmax Activation Function
    u = z - np.max(z, axis=-1, keepdims=True)
    num = np.exp(u)
    denom = np.sum(num, axis=-1, keepdims=True)
    return num / denom

def relu(z): #Relu Activation Function
    return np.maximum(0, z)

def tanh(z): #Tanh Activation Function
    return np.tanh(z)

"""#Gradients of Activation Functions"""

def grad_sigmoid(z): #Gradient of Sigmoid Activation Function
    return sigmoid(z) * (1 - sigmoid(z))

def grad_softmax(z): #Gradient of Softmax Activation Function
    s = softmax(z)
    return s * (1 - s)

def grad_relu(z): #Gradient of Relu Activation Function
    return 1 * (z > 0)

def grad_tanh(z): #Gradient of Tanh Activation Function
    return 1 - (np.tanh(z) ** 2)