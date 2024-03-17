import numpy as np
import copy
import math

"""#Gradient Descent Variation"""

#Basic gradient descent class
class basic_gradient_descent:
    def __init__(self, eta, num_layers, weight_decay=0.0):
        self.eta = eta #Learning rate in GD algorithm
        self.num_layers = num_layers
        self.calls = 1 #Number of calls made, to adapt learning rate
        self.l_control = 1.0 #To control learning rate
        self.weight_decay = weight_decay

    #Update in Gradient Descent
    def grad_update(self, neural_network, gradient, m):
        for i in range(self.num_layers):
            #Updating weights and biases
            neural_network[i]['weight'] = neural_network[i]['weight'] - ((self.eta / self.l_control) * gradient[i]['weight']) - (self.eta * self.weight_decay * neural_network[i]['weight'])/m
            neural_network[i]['bias'] = neural_network[i]['bias'] - ((self.eta / self.l_control) * gradient[i]['bias'])/m
        self.calls += 1
        if self.calls % 10 == 0: #Shrinking learning rate after every 10 steps to ensure it don't miss minima and diverge
            self.l_control += 1.0


#Momentum based gradient descent class
class momentum_gradient_descent:
    def __init__(self, eta, num_layers, beta, weight_decay=0.0):
        self.eta = eta #Learning rate in GD algorithm
        self.beta = beta #Momentum rate
        self.num_layers = num_layers
        self.calls = 1 #Number of calls made, to adapt learning rate
        self.l_control = 1 #To control learning rate

        self.momentum = None #Momentum push depending on previous updates
        self.weight_decay = weight_decay

    #Update in Gradient Descent
    def grad_update(self, neural_network, gradient, m):
        """
        Sutskever, I. & Martens, J. & Dahl, G. & Hinton, G.. (2013).
        On the importance of initialization and momentum in deep learning. 30th International Conference on Machine Learning, ICML 2013. 1139-1147.
        """
        beta = min(1 - 2 ** (-1 - math.log((self.calls / 250.0) + 1, 2)), self.beta) #Adaptive momentum as suggested by Sutskever et. al., 2013
        if self.momentum is None:
            self.momentum = copy.deepcopy(gradient) #Copying the structure
            #Initialization of momentum
            for i in range(self.num_layers):
                self.momentum[i]['weight'] = (self.eta / self.l_control) * gradient[i]['weight']/m
                self.momentum[i]['bias'] = (self.eta / self.l_control) * gradient[i]['bias']/m
        else:
            #Update in momentum
            for i in range(self.num_layers):
                self.momentum[i]['weight'] = beta * self.momentum[i]['weight'] + (self.eta / self.l_control) * gradient[i]['weight']/m
                self.momentum[i]['bias'] = beta * self.momentum[i]['bias'] + (self.eta / self.l_control) * gradient[i]['bias']/m
        #Updates depending on history and present component
        for i in range(self.num_layers):
            #Updating weight and biases
            neural_network[i]['weight'] = neural_network[i]['weight'] - self.momentum[i]['weight'] - ((self.eta / self.l_control) * self.weight_decay * neural_network[i]['weight'])/m
            neural_network[i]['bias'] = neural_network[i]['bias'] - self.momentum[i]['bias']

        self.calls += 1
        if self.calls % 10 == 0: #Shrinking learning rate after every 10 steps to ensure it don't miss minima and diverge
            self.l_control += 1.0


#Nesterov Accelerated gradient descent class
class nestrov_gradient_descent:
    def __init__(self, eta, num_layers, beta, weight_decay=0.0):
        self.eta = eta #Learning rate in GD algorithm
        self.beta = beta #Momentum rate
        self.num_layers = num_layers
        self.calls = 1 #Number of calls made, to adapt learning rate
        self.l_control = 1 #To control learning rate

        self.momentum = None #Momentum push depending on previous updates
        self.weight_decay = weight_decay

    #Lookahead point based on history movement
    def history_movement(self, neural_network):
        if self.momentum is None:  #As momentum begin only when we have some past movements
            pass
        else:
            for i in range(self.num_layers):
                neural_network[i]['weight'] -= self.beta * self.momentum[i]['weight']
                neural_network[i]['bias'] -= self.beta * self.momentum[i]['bias']

    #Update in Gradient Descent
    def grad_update(self, neural_network, gradient, m):
        for i in range(self.num_layers):
            #Updating weights and biases
            neural_network[i]['weight'] = neural_network[i]['weight'] - ((self.eta / self.l_control) * gradient[i]['weight'])/m - ((self.eta / self.l_control) * self.weight_decay * neural_network[i]['weight'])/m
            neural_network[i]['bias'] -= self.eta * gradient[i]['bias']/m

        beta = min(1 - 2 ** (-1 - math.log((self.calls / 250.0) + 1, 2)), self.beta) #Adaptive momentum as suggested by Sutskever et. al., 2013

        if self.momentum is None:
            self.momentum = copy.deepcopy(gradient) #Copying the structure
            for i in range(self.num_layers):
                self.momentum[i]['weight'] = (self.eta / self.l_control) * gradient[i]['weight']/m
                self.momentum[i]['bias'] = (self.eta / self.l_control) * gradient[i]['bias']/m
        else:
            #Update in momentum
            for i in range(self.num_layers):
                self.momentum[i]['weight'] = beta * self.momentum[i]['weight'] + ((self.eta / self.l_control) * gradient[i]['weight'])/m
                self.momentum[i]['bias'] = beta * self.momentum[i]['bias'] + ((self.eta / self.l_control) * gradient[i]['bias'])/m

        self.calls += 1
        if self.calls % 10 == 0: #Shrinking learning rate after every 10 steps to ensure it don't miss minima and diverge
            self.l_control += 1.0


#RMSprop gradient descent class
class RMSprop_gradient_descent:
    def __init__(self, eta, num_layers, beta, weight_decay=0.0, epsilon = 1e-3):
        self.eta = eta #Learning rate in GD algorithm
        self.beta = beta #Decay rate for moving average
        self.num_layers = num_layers
        self.calls = 1 #Number of calls made. No need to adapt learning rate in this algorithm.
        self.epsilon = epsilon

        self.update = None
        self.weight_decay = weight_decay

    #Update in Gradient Descent
    def grad_update(self, neural_network, gradient, m):
        if self.update is None:
            self.update = copy.deepcopy(gradient) #Copying the structure
            for i in range(self.num_layers):
                self.update[i]['weight'] = (1 - self.beta) * (gradient[i]['weight']/m) ** 2
                self.update[i]['bias'] = (1 - self.beta) * (gradient[i]['bias']/m) ** 2
        else:
            for i in range(self.num_layers):
                self.update[i]['weight'] = self.beta * self.update[i]['weight'] + (1 - self.beta) * (gradient[i]['weight']/m) ** 2
                self.update[i]['bias'] = self.beta * self.update[i]['bias'] + (1 - self.beta) * (gradient[i]['bias']/m) ** 2
        #Update rule for RMSProp
        for i in range(self.num_layers):
            #updating weight and biases
            neural_network[i]['weight'] = neural_network[i]['weight'] - np.multiply((self.eta / np.sqrt(self.update[i]['weight'] + self.epsilon)),gradient[i]['weight']/m) - self.eta*self.weight_decay * neural_network[i]['weight']/m
            neural_network[i]['bias'] = neural_network[i]['bias'] - np.multiply((self.eta / np.sqrt(self.update[i]['bias'] + self.epsilon)), gradient[i]['bias']/m)

        self.calls += 1


#Adam gradient descent class
class adam_gradient_descent:
    def __init__(self, eta, num_layers, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.eta = eta #Learning rate in GD algorithm
        self.beta1 = beta1 #Decay rate for moving average of first moments
        self.beta2 = beta2 #Decay rate for moving average of second moments
        self.num_layers = num_layers
        self.calls = 1 #Number of calls made. No need to adapt learning rate in this algorithm.

        self.momentum = None #First moment vectors initialized to None, as momentum is when we have past movements.
        self.momentum_hat = None

        self.second_momentum = None #Second moment vectors initialized to None, as momentum is when we have past movements.
        self.second_momentum_hat = None

        self.epsilon = epsilon
        self.weight_decay = weight_decay

    #Update in Gradient Descent
    def grad_update(self, neural_network, gradient,m):
        if self.momentum is None:
            self.momentum = copy.deepcopy(gradient) #Copying the structure
            self.second_momentum = copy.deepcopy(gradient)
            for i in range(self.num_layers):
                self.momentum[i]['weight'][:] = np.zeros_like(gradient[i]['weight'])
                self.momentum[i]['bias'][:] = np.zeros_like(gradient[i]['bias'])
                self.second_momentum[i]['weight'][:] = np.zeros_like(gradient[i]['weight'])
                self.second_momentum[i]['bias'][:] = np.zeros_like(gradient[i]['bias'])
            self.momentum_hat = copy.deepcopy(self.momentum) #Copying the structure
            self.second_momentum_hat = copy.deepcopy(self.second_momentum)

        for i in range(self.num_layers):
            #Updating the first and moment estimate
            self.momentum[i]['weight'] = self.beta1 * self.momentum[i]['weight'] + (1 - self.beta1) * gradient[i]['weight']/m
            self.momentum[i]['bias'] = self.beta1 * self.momentum[i]['bias'] + (1 - self.beta1) * gradient[i]['bias']/m

            self.second_momentum[i]['weight'] = self.beta2 * self.second_momentum[i]['weight'] + (1 - self.beta2) * np.power(gradient[i]['weight']/m, 2)
            self.second_momentum[i]['bias'] = self.beta2 * self.second_momentum[i]['bias'] + (1 - self.beta2) * np.power(gradient[i]['bias']/m, 2)

        for i in range(self.num_layers):
            self.momentum_hat[i]['weight'][:] = (1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['weight']
            self.momentum_hat[i]['bias'][:] = (1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['bias']

            self.second_momentum_hat[i]['weight'][:] = (1 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i]['weight']
            self.second_momentum_hat[i]['bias'][:] = (1 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i]['bias']

        for i in range(self.num_layers):
            #Updating weight and biases
            denom_w = 1/(np.sqrt(self.second_momentum_hat[i]['weight']) + self.epsilon)
            neural_network[i]['weight'] = neural_network[i]['weight'] - self.eta * (np.multiply(denom_w, self.momentum_hat[i]['weight'])) - (self.eta * self.weight_decay * neural_network[i]['weight'])/m

            denom_b = 1/(np.sqrt(self.second_momentum_hat[i]['bias'])+ self.epsilon)
            neural_network[i]['bias'] -= self.eta * np.multiply(denom_b, self.momentum_hat[i]['bias'])

        self.calls += 1

#Nadam gradient descent class
class nadam_gradient_descent:
    def __init__(self, eta, num_layers, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.eta = eta #Learning rate in GD algorithm
        self.beta1 = beta1 #Decay rate for moving average of first moments
        self.beta2 = beta2 #Decay rate for moving average of second moments
        self.num_layers = num_layers
        self.calls = 1 #Number of calls made. No need to adapt learning rate in this algorithm.

        self.momentum = None #First moment vector
        self.momentum_hat = None

        self.second_momentum = None #Second moment vector
        self.second_momentum_hat = None

        self.epsilon = epsilon
        self.weight_decay = weight_decay

    #Update in Gradient Descent
    def grad_update(self, neural_network, gradient, m):
        if self.momentum is None:
            self.momentum = copy.deepcopy(gradient) #Copying the structure
            self.second_momentum = copy.deepcopy(gradient)
            for i in range(self.num_layers):
                self.momentum[i]['weight'] = (1 - self.beta1) * gradient[i]['weight']/m
                self.momentum[i]['bias'] = (1 - self.beta1) * gradient[i]['bias']/m

                self.second_momentum[i]['weight'] = (1 - self.beta2) * np.power(gradient[i]['weight']/m, 2)
                self.second_momentum[i]['bias'] = (1 - self.beta2) * np.power(gradient[i]['bias']/m, 2)
        else:
            for i in range(self.num_layers):
                #Updating the first and second moment estimates
                self.momentum[i]['weight'] = self.beta1 * self.momentum[i]['weight'] + (1 - self.beta1) * gradient[i]['weight']/m
                self.momentum[i]['bias'] = self.beta1 * self.momentum[i]['bias'] + (1 - self.beta1) * gradient[i]['bias']/m

                self.second_momentum[i]['weight'] = self.beta2 * self.second_momentum[i]['weight'] + (1 - self.beta2) * np.power(gradient[i]['weight']/m, 2)
                self.second_momentum[i]['bias'] = self.beta2 * self.second_momentum[i]['bias'] + (1 - self.beta2) * np.power(gradient[i]['bias']/m, 2)

        momentum_hat = copy.deepcopy(self.momentum) #Copying the structure
        second_momentum_hat = copy.deepcopy(self.second_momentum)
        for i in range(self.num_layers):
            momentum_hat[i]['weight'] = (self.beta1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['weight'] + ((1 - self.beta1) / (1 - (self.beta1 ** self.calls))) * gradient[i]['weight']
            momentum_hat[i]['bias'] = (self.beta1 / (1 - (self.beta1 ** self.calls))) * self.momentum[i]['bias'] + ((1 - self.beta1) / (1 - (self.beta1 ** self.calls))) * gradient[i]['bias']

            second_momentum_hat[i]['weight'] = (self.beta2 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i]['weight']
            second_momentum_hat[i]['bias'] = (self.beta2 / (1 - (self.beta2 ** self.calls))) * self.second_momentum[i]['bias']

        for i in range(self.num_layers):
            #Updating weight and biases
            denom_w = 1/(np.sqrt(self.second_momentum[i]['weight'] + self.epsilon))
            neural_network[i]['weight'] = neural_network[i]['weight'] - self.eta * (np.multiply(denom_w, momentum_hat[i]['weight'])) - (self.eta * self.weight_decay * neural_network[i]['weight'])/m

            denom_b = 1/(np.sqrt(self.second_momentum[i]['bias']) + self.epsilon)
            neural_network[i]['bias'] -= self.eta * np.multiply(denom_b, second_momentum_hat[i]['bias'])

        self.calls += 1