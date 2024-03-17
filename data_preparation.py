import numpy as np
from keras.datasets import fashion_mnist, mnist

def dataset(data):
  #Download, Normalize and Split the dataset into train and validation
  #In this function, we doing it to MNIST and Fashion-MNIST Dataset.
  if data == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'] #Labels in Fashion-MNIST Dataset, i.e. list of 10 fashion items
  if data =='mnist':
    (train_images, train_labels),(test_images, test_labels)=mnist.load_data()
    labels = list(range(10)) #Labels in MNIST Dataset, i.e. digits from 0 to 9
  dim = train_images.shape[1]**2
  train_images = np.reshape(train_images, [-1, dim]) #Reshaping it to 1-D array to input in neural network
  test_images = np.reshape(test_images, [-1, dim])

  #Normalizing the dataset
  train_images = train_images.astype(np.float64)
  train_images = (train_images - np.min(train_images))/(np.max(train_images)-np.min(train_images))

  test_images = test_images.astype(np.float64)
  test_images = (test_images - np.min(test_images))/(np.max(test_images)-np.min(test_images))

  num_classes = len(labels) #Number of labels

  #Splitting into train and validation set
  validation_split_percentage = 10  # 10% of the dataset should go to the validation set
  total_datapoints = train_images.shape[0]
  validation_size = int((validation_split_percentage / 100) * train_images.shape[0])

  # Calculate the size of the training set
  training_size = total_datapoints - validation_size

  # Generate indices for the dataset
  indices = np.arange(total_datapoints)

  # Shuffle indices to ensure random distribution
  np.random.shuffle(indices)

  # Split indices into training and validation sets
  train_indices = indices[:training_size]
  valid_indices = indices[training_size:]

  valid_images = train_images[valid_indices]
  train_images = train_images[train_indices]

  valid_labels = train_labels[valid_indices]
  train_labels = train_labels[train_indices]

  train_labels = np.eye(num_classes)[train_labels]
  valid_labels = np.eye(num_classes)[valid_labels]
  test_labels = np.eye(num_classes)[test_labels]

  return train_images, valid_images, test_images, train_labels, valid_labels, test_labels, labels