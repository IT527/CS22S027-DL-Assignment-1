# CS22S027-DL-Assignment-1
CS6910: Deep Learning Assignmnet 1, Jan-May 2024

This project is an implementation of a neural network from scratch using python. It is designed to be flexible, allowing adjustments to various parameters such as dataset selection (mnist and Fashion-mnist), network architecture, activation functions, and experiment tracking using wandb.


### Dependencies
 - python
 - numpy library
 - wandb library
 - copy library
 - tqdm library (for fast, extensible progress bar for loops and iterations in python)
 - joblib library (for efficiently dealing with large numpy arrays)
 - tensorflow and keras library (for downloading mnist and fashion_mnist dataset)
 - matplotlib (If you want to plot confusion matrix)

To download all the dependencies, you can run: `pip install -r requirements.txt`


### Clone and Download Instructions
Clone the repository or download the project files. Ensure that python and other required packages are installed in the project directory.
To clone the repository directly to you local machine, ensure git is installed, run the command: 
</br>
`git clone https://github.com/IT527/CS22S027-DL-Assignment-1.git`
</br>
Alternatively, you can download the entire repository as a zip file from the Download ZIP option provided by github.


### Usage
To run the python script, navigate to the project directory and run: `python train.py [OPTIONS]`
The 'OPTIONS' can take different values for parameters to select dataset, modify network architecture, select activation function and many more.
The possible arguments and respective values for 'OPTIONS' are shown in the table below:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | cs22s027  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 128 | Batch size used to train neural network. | 
| `-l`, `--loss` | "cross_entropy" | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | "adam" | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.99 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.00000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0.5 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | "xavier" | choices:  ["random", "xavier"] | 
| `-nhl`, `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["sigmoid", "tanh", "relu"] |


An example run with dataset "mnist" and number of hidden layers as 5 : `python train.py --dataset "mnist" --num_layers 5`

</br>

On execution of the file as shown above, loss and accuracies for the train, validation and test dataset will be printed on the terminal. Along with it, the plots highlighting the loss and accuracies for each epochs, for both train and validation dataset, will be logged onto the wandb project.
To access plots in wandb, ensure to replace the given key with your wandb API key.
Look for line 14 in train.py file and enter your API key in the key variable.


### Additional Resources and help
Included in the project is DL_Assignment_1_.ipynb, compatible with Jupyter Notebook or Google Colab. It encompasses neural network codes, sweep operations, and logging utilities like confusion matrices and dataset images. For tailored runs, you may need to adjust configurations and uncomment sections in the notebook to log specific metrics or plots. The notebook serves as a practical reference for understanding the project's workflow. 
All the plots are generated and logged to wandb using this file only, whle for a new configuration one can run the train.py file as shown above.


The sweep details for choosing the hyperparameters, runs, sample images, and related plots can be viewed at: ``



