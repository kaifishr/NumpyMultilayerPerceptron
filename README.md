# NumpyMultilayerPerceptron

This implementation of a Multilayer Perceptron network was written in Python using only Numpy. The network can process any input represented as an array. Here, the performance of this implementation is checked using the MNIST and Fashion-MNIST dataset. The weights and biases of the network are optimized using the simplest form of stochastic gradient descent.

The network takes the following parameters for training `learning_rate`, `batch_size`, `epochs`. The network's architecture (depth and width) is defined by `layer_size` which represents a list of integers. Here, the following training parameters are used:

```python
# Training parameters
learning_rate = 0.1
batch_size = 64
epochs = 100

# Network architecture
layer_size = [n_input, 128, 128, 128, n_classes]
````
Here are some results from a short run of 100 epochs for both, the MNIST and Fashion-MNIST dataset. After 100 epochs, the network with three hidden layers of size 256 achieved the highest accuracy of 98.09% for the MNIST and 89.69% for the Fashion-MNIST dataset.

The resulting graphs show the loss and accuracy for the evaluation dataset. In case of the Fashion-MNIST dataset the plots, loss and accuracy show tendencies of over-fitting. This shows, that the network starts to remember the training data and performes worse with new data such as the evaluation and test dataset.

<div align="center">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/mnist_eval_loss.png" height="340">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/mnist_eval_accuracy.png" height="340">
</div>

<div align="center">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/fmnist_eval_loss.png" height="480">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/fmnist_eval_accuracy.png" height="480">
</div>

The weights that connect the input with the first hidden layer can be visualized and can help to better understand what the network learned during training.

<div align="center">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/mnist_weights.png" height="224" width="224">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/fashion_mnist_weights.png" height="224" width="224">
</div>
