import numpy as np
import copy
import math
from tqdm import tqdm
import joblib
import random
from activation import *
from arguments import *
from data_preparation import *
from optimiser import *
import wandb

args = parsArg()
key = '90512e34eff1bdddba0f301797228f7b64f546fc'
data = args.dataset
num_layers = args.num_layers
num_layers = num_layers + 2 #Input and Output
hidden_layer_sizes = args.hidden_size
activation = args.activation
weight_init = args.weight_init
epochs = args.epochs
weight_decay = args.weight_decay
optimiser = args.optimizer
eta = args.learning_rate
batch = args.batch_size
loss_type = args.loss

wandb.login(key = key)

"""#Weight Initialization"""

#To add a layer in Neural Network, the first layer will be input and last being output.
def hidden_layer(num_neurons, wt_init, activation, dim= None):
  layer = {} #Declaring an empty dictionary to store values of respective weights, bias and activation functions.
  if dim is not None: #Need to ensure that structure is valid. It is necessary for matrix multiplications to hold.
    if wt_init == 'random':
      layer['weight'] = np.random.normal(0,0.005, (num_neurons, dim))
      layer['bias'] = np.random.normal(0,0.005, (num_neurons, 1))
    elif wt_init == 'xavier':
      limit = limit = np.sqrt(2 / float(dim + num_neurons))
      layer['weight'] = np.random.normal(0, limit, (num_neurons, dim))
      layer['bias'] = np.random.normal(0,limit, (num_neurons, 1))
  else:
    previous_num_neurons = neural_network[-1]['h'].shape[0]
    if wt_init == 'random':
      layer['weight'] = np.random.normal(0,0.005, (num_neurons, previous_num_neurons))
      layer['bias'] = np.random.normal(0,0.005, (num_neurons, 1))
    elif wt_init == 'xavier':
      imit = limit = np.sqrt(2 / float(previous_num_neurons + num_neurons))
      layer['weight'] = np.random.normal(0, limit, (num_neurons, previous_num_neurons))
      layer['bias'] = np.random.normal(0,limit, (num_neurons, 1))
  layer['h'] = np.zeros((num_neurons, 1)) #Initializing to zeroes intially.
  layer['a'] = np.zeros((num_neurons, 1)) #Initializing to zeroes intially.
  layer['activation'] = activation #Setting activation function for added layer
  neural_network.append(layer)


"""#Loss Functions"""

def cross_entropy(y, y_hat_vector, L):   #y = true label and y_hat_vector = softmax output
    m = y.shape[0]  # Number of samples

    # L2 Regularization Loss
    L2_regularization_loss = 0
    for i in range(1, L):  # Start from 1 as index 0 is for the input layer of neural network
        L2_regularization_loss += np.sum(np.square(neural_network[i]['weight']))
    L2_regularization_loss = L2_regularization_loss/ (2 * m)

    # Cross-Entropy Loss
    cross_entropy_loss = -np.sum(y * np.log(y_hat_vector + 1e-9)) / m

    # Total Loss is the sum of above two quantities
    total_loss = cross_entropy_loss + L2_regularization_loss

    return total_loss

def squared_loss(y, y_hat_vector, L):     #y = true label and y_hat_vector = softmax output
  m = y.shape[0]
  #L2 Regularization Loss
  L2_regularization_loss = 0
  for i in range(1, L):  # Start from 1 as index 0 is for input layer
    L2_regularization_loss += np.sum(np.square(neural_network[i]['weight']))
  L2_regularization_loss = L2_regularization_loss/ (2 * m)
  #Mean-squared error loss
  mse_loss = (np.linalg.norm(y-y_hat_vector)**2)/m

  #Total Loss is the sum of above two quantities
  total_loss = mse_loss + L2_regularization_loss

  return total_loss

"""# Loss and Accuracy"""

def loss_acc_calc(L, X, y, loss_type):
    loss = 0
    acc = 0
    if loss_type == 'cross_entropy':
        forward_propagation(L, X)
        loss = cross_entropy(y, neural_network[L - 1]['h'], L)
        y_pred = np.argmax(neural_network[L - 1]['h'], axis=1)
        y_true = np.argmax(y, axis=1)
        acc = np.sum(y_pred == y_true)/y_true.shape[0]

    elif loss_type == 'mean_squared_error':
        forward_propagation(L, X)
        loss = squared_loss(y, neural_network[L - 1]['h'],L)
        y_pred = np.argmax(neural_network[L - 1]['h'], axis=1)
        y_true = np.argmax(y, axis=1)
        acc = np.sum(y_pred == y_true)/y_true.shape[0]
    return [loss, acc]

"""# Forward Propagation"""

def forward_propagation(L, X):
    for i in range(L):
        if i == 0:  #We have input layer, X, as our first layer
            neural_network[i]['a'] = np.dot(X, neural_network[i]['weight'].T) + neural_network[i]['bias'].T
        else:
            neural_network[i]['a'] = np.dot(neural_network[i-1]['h'], neural_network[i]['weight'].T) + neural_network[i]['bias'].reshape(-1)

        if neural_network[i]['activation'] == 'sigmoid':
          neural_network[i]['h'] = sigmoid(neural_network[i]['a'])
        elif neural_network[i]['activation'] == 'softmax':
          neural_network[i]['h'] = softmax(neural_network[i]['a'])
        elif neural_network[i]['activation'] == 'relu':
          neural_network[i]['h'] = relu(neural_network[i]['a'])
        elif neural_network[i]['activation'] == 'tanh':
          neural_network[i]['h'] = tanh(neural_network[i]['a'])

"""# Back Propagation"""

def back_propagation(L, X, y, loss_type):
  if loss_type == 'cross_entropy':
    gradient[L-1]['a'] = neural_network[L-1]['h'] - y
  if loss_type == 'mean_squared_error':
    gradient[L-1]['a'] = np.multiply((neural_network[L-1]['h']-y), grad_softmax(neural_network[L-1]['a']))
  for i in range(L-1, 0, -1):
    gradient[i]['weight'] = np.matmul(gradient[i]['a'].T, (neural_network[i-1]['h']))
    gradient[i]['bias'] = np.sum(gradient[i]['a'], axis = 0).reshape(-1,1)
    gradient[i-1]['h'] = np.matmul(gradient[i]['a'], neural_network[i]['weight'])

    if neural_network[i-1]['activation'] == 'sigmoid':
      gradient[i-1]['a'] = np.multiply(gradient[i-1]['h'], grad_sigmoid(neural_network[i-1]['a']))
    elif neural_network[i-1]['activation'] == 'softmax':
      gradient[i-1]['a'] = np.multiply(gradient[i-1]['h'], grad_softmax(neural_network[i-1]['a']))
    elif neural_network[i-1]['activation'] == 'relu':
      gradient[i-1]['a'] = np.multiply(gradient[i-1]['h'], grad_relu(neural_network[i-1]['a']))
    elif neural_network[i-1]['activation'] == 'tanh':
      gradient[i-1]['a'] = np.multiply(gradient[i-1]['h'], grad_tanh(neural_network[i-1]['a']))

X_train, X_valid, X_test, y_train, y_valid, y_test, labels = dataset(data)

"""#Fitting Data"""

def train_and_evaluate(X_train, X_valid, y_train, y_valid, batch, epochs, opt, loss_type):
    L = len(neural_network)  #Number of layers in neural network including input and output layer
    #Loop for epoch iterations
    for k in tqdm(range(epochs)):
        for i in range(int(X_train.shape[0]/batch)):
            global loss
            loss = 0
            indices = np.random.choice(range(X_train.shape[0]), batch) #Sample indices of a batch
            x_batch = X_train[indices]
            y_batch = y_train[indices]
            if isinstance(opt, nestrov_gradient_descent): #For Nestrov gradient descent, we go to lookahead point and then compute gradient
                opt.history_movement(neural_network=neural_network)

            forward_propagation(L, x_batch)
            back_propagation(L, x_batch, y_batch, loss_type=loss_type)
            opt.grad_update(neural_network=neural_network, gradient=gradient, m = x_batch.shape[0])

        #Computing loss and accuracy on train and validation set of datapoints
        validation_result = loss_acc_calc(L, X= X_valid, y= y_valid,loss_type=loss_type)
        training_result = loss_acc_calc(L, X=X_train, y=y_train, loss_type=loss_type)
        
        #Logging to WandB
        wandb.log({"val_accuracy": validation_result[1], 'val_loss': validation_result[0],
                   'train_accuracy': training_result[1], 'train_loss': training_result[0], 'epoch': k + 1})

        if np.isnan(validation_result[0]):
            return
    print("Train Accuracy = ", str(training_result[1]*100), "%")
    print("Train Loss = ", str(training_result[0]))

    print("Validation Accuracy = ", str(validation_result[1]*100), "%")
    print("Validation Loss = ", str(validation_result[0]))

"""# Initializing Neural Network"""

def initialize_fit_model(batch, epochs, output_dim, activation, opt, num_layers, hidden_layer_sizes, weight_init='xavier',loss_type='cross_entropy'):

    n_features = 784
    global neural_network
    global gradient
    neural_network = []
    gradient = []

    selected_sizes = np.full(num_layers-1, hidden_layer_sizes)

    for i, layer_size in enumerate(selected_sizes):
        if i == 0:
            # For the first layer, specify the input dimension
            hidden_layer(num_neurons= layer_size, activation=activation, dim=784, wt_init=weight_init)
        else:
            # Subsequent layers infer their input dimension from the previous layer's output
            hidden_layer(num_neurons= layer_size, activation=activation, wt_init=weight_init)

    # Adding the output layer
    hidden_layer(num_neurons=output_dim, activation='softmax', wt_init=weight_init)

    """Replicate the structure of neural_network."""
    gradient = copy.deepcopy(neural_network)

    train_and_evaluate(X_train, X_valid, y_train, y_valid, batch=batch, epochs=epochs, opt=opt,loss_type=loss_type)
    return neural_network

def network_train():
    #Best Configuration by default
    config={"batch_size" :batch, "epoch":epochs, "activation":activation, "optimiser":optimiser, "num_layers" : num_layers, "hidden_layer_sizes" :hidden_layer_sizes ,"weight_init":weight_init,"loss":loss_type,"learning_rate":	eta, "weight_decay":weight_decay}
    run = wandb.init(config = config, project = args.wandb_project,entity=args.wandb_entity)
    opti = None
    wandb.run.name = 'hln' + str(run.config.num_layers) + 'hls' + str(run.config.hidden_layer_sizes) + 'bs_' + str(run.config.batch_size) + '_act_' + run.config.activation + '_opt_' + str(
        run.config.optimiser) + '_ini_' + str(run.config.weight_init) + '_epoch' + str(run.config.epoch) + '_lr_' + str(
        round(run.config.learning_rate, 4)) + 'wd' + str(run.config.weight_decay) #Naming each run in wandb, to ensure parameter values are understandable from the respective run name.
    if run.config.optimiser == 'nag':
        opti = nestrov_gradient_descent(num_layers = run.config.num_layers, eta=run.config.learning_rate, beta=.90, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'rmsprop':
        opti = RMSprop_gradient_descent(num_layers = run.config.num_layers, eta=run.config.learning_rate, beta=.90, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'sgd':
        opti = basic_gradient_descent(num_layers = run.config.num_layers, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'momentum':
        opti = momentum_gradient_descent(num_layers = run.config.num_layers, eta=run.config.learning_rate, beta=.99,
                                       weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'adam':
        opti = adam_gradient_descent(num_layers = run.config.num_layers, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)
    elif run.config.optimiser == 'nadam':
        opti = nadam_gradient_descent(num_layers = run.config.num_layers, eta=run.config.learning_rate, weight_decay=run.config.weight_decay)

    final_neural_network=initialize_fit_model(epochs=run.config.epoch, batch=run.config.batch_size, output_dim=10,
           opt=opti, weight_init=run.config.weight_init, activation=run.config.activation, num_layers = run.config.num_layers, hidden_layer_sizes = run.config.hidden_layer_sizes , loss_type=run.config.loss)

    return final_neural_network

network=network_train()

def test_prediction(network):
  L = len(network)
  test_result = loss_acc_calc(L, X_test, y_test, "cross_entropy") #Computing test loss and test accuracy on test dataset
  y_pred = np.argmax(network[L - 1]['h'], axis=1)
  loss = np.round(test_result[0],4) #Rounding loss to 4 decimal place
  accuracy = np.round(test_result[1]*100,2) #Rounding accuracy to 2 decimal place
  return y_pred, loss, accuracy


y_pred, loss, accuracy = test_prediction(network)
print("Test Accuracy = ", str(accuracy), "%")
print("Test Loss = ", str(loss))
