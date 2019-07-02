# NumpyMultilayerPerceptron

This implementation of a Multilayer Perceptron network was written in Python using Numpy. The network can process any input represented as an array. Here, the performance of this implementation is checked using the MNIST and Fashion-MNIST dataset. These datasets can be downloaded [here (MNIST)](http://yann.lecun.com/exdb/mnist/) and [here (Fashion-MNIST)](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion). The weights and biases of the network are optimized using the simplest form of stochastic gradient descent.

The network takes the following parameters for training `learning_rate`, `batch_size`, `epochs`. The network's architecture (depth and width) is defined by `layer_size` which represents a list of integers. Here, the following training parameters are used:

```python
# Training parameters
learning_rate = 0.1
batch_size = 64
epochs = 100

# Network architecture
network_config = [n_input, 128, 128, 128, n_classes]
````

Here are some results for a three-layered network with 64, 128 and 256 neurons per layer trained for both, the MNIST and Fashion-MNIST dataset and 100 epochs. The learning rate was set to 0.1 and a batch size of 64 images was used. The largest network achieved the highest accuracy of 98.09% for the MNIST and 89.69% for the Fashion-MNIST dataset.

The resulting graphs show the loss and accuracy for the evaluation dataset. In case of the Fashion-MNIST dataset loss and accuracy show tendencies of over-fitting. This shows that the network does not generalize very well to unseen data because it remembers the training data but performs worse with new data such as the evaluation and test dataset.

**MNIST:**
<div align="center">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/results/mnist_eval_loss.png" height="320">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/results/mnist_eval_accuracy.png" height="320">
</div>

**Fashion-MNIST:**
<div align="center">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/results/fmnist_eval_loss.png" height="320">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/results/fmnist_eval_accuracy.png" height="320">
</div>

The weights that connect the input with the first hidden layer can be visualized and can help to better understand what the network learned during training. The weights of the network trained on the MNIST dataset are shown on the left. Weights for the Fashion-MNIST dataset are shown on the right.

<div align="center">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/results/mnist_weights.png" height="320" width="320">
<img src="https://github.com/KaiFabi/NumpyMultilayerPerceptron/blob/master/results/fashion_mnist_weights.png" height="320" width="320">
</div>
