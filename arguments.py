import argparse

def parsArg():
    parser = argparse.ArgumentParser(description="Train a neural network.")

    # Add arguments to the parser
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', type=str, help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='cs22s027', type=str, help='WandB Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', type=str, choices=['mnist', 'fashion_mnist'], help='Dataset to use.')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='Batch size used to train neural network.')
    parser.add_argument('-l', '--loss', default='cross_entropy', type=str, choices=['mean_squared_error', 'cross_entropy'], help='Loss function to use.')
    parser.add_argument('-o', '--optimizer', default='adam', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer to use.')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='Learning rate used to optimize model parameters.')
    parser.add_argument('-m', '--momentum', default=0.5, type=float, help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('--beta', default=0.9, type=float, help='Beta used by rmsprop optimizer.')
    parser.add_argument('--beta1', default=0.9, type=float, help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('--beta2', default=0.99, type=float, help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('--epsilon', default=0.000001, type=float, help='Epsilon used by optimizers.')
    parser.add_argument('-wd', '--weight_decay', default=0.5, type=float, help='Weight decay used by optimizers.')
    parser.add_argument('-wi', '--weight_init', default='xavier', type=str, choices=['random', 'xavier'], help='Weight initialization method.')
    parser.add_argument('-nhl', '--num_layers', default=3, type=int, help='Number of hidden layers used in feedforward neural network.')
    parser.add_argument('-sz', '--hidden_size', default=128, type=int, help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a', '--activation', default='tanh', type=str, choices=['sigmoid', 'tanh', 'relu'], help='Activation function to use.')

    # Parse the arguments
    args = parser.parse_args()
    return args