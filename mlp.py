import os
import data
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit # numerically stable sigmoid function

# Seed
np.random.seed(seed=73214)

# Activation functions
def sigma(x):
    return expit(x)

def sigma_prime(x):
    u = sigma(x)
    return u-u*u

def relu(x):
    return np.maximum(x, 0.)

def relu_prime(x):
    return (x>0).astype(x.dtype)

# Xavier weight initialization
def xavier_init(in_size, out_size):
    return np.sqrt(6./(in_size+out_size))

# Multilayer Perceptron Class
class NeuralNetwork(object):

    def __init__(self, network_config):

        self.network_depth = len(network_config)
        self.network_config = network_config

        # Weights 
        self.W = [np.random.uniform(-xavier_init(network_config[l],network_config[l+1]), \
                xavier_init(network_config[l],network_config[l+1]), \
                size=(network_config[l+1],network_config[l])) for l in range(self.network_depth-1)]
        # Bias
        self.b = [0.1*np.ones((network_config[l],1)) for l in range(1,self.network_depth)]
        # Pre-activation
        self.z = [np.zeros((self.network_config[l],1)) for l in range(1,self.network_depth)]
        # Activations
        self.a = [np.zeros((self.network_config[l],1)) for l in range(self.network_depth)]
        # Gradients
        self.dW = [np.zeros((self.network_config[l+1],self.network_config[l])) for l in range(self.network_depth-1)] 
        self.db = [np.zeros((self.network_config[l],1)) for l in range(1,self.network_depth)]


    def grouped_rand_idx(self, n_total, batch_size):
        idx = np.random.permutation(n_total)
        return [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]


    def optimize(self, x_train, y_train, x_eval, y_eval, x_test, y_test, epochs, batch_size, learning_rate):
         
        eta = learning_rate/batch_size
        for epoch in range(epochs):
            idx_list = self.grouped_rand_idx(len(x_train), batch_size)
            for idx in idx_list:
                # Get batch of random training samples
                x_batch, y_batch = x_train[idx], y_train[idx]
                # Feedforward pass
                self.feedforward(x_batch) 
                # Backpropapagation with gradient descent
                self.backprop_gradient_descent(y_batch, eta)
 
            if epoch % 1 == 0:
                self.prediction(x_eval, y_eval, epoch, mode="eval")

            if epoch % 30 == 0:
                self.visualize_weights(epoch)

        # Compute test accuracy and loss
        self.prediction(x_test, y_test, epoch, mode="test")


    def backprop_gradient_descent(self, Y, eta):
        # Backpropagation
        delta = (self.a[-1] - Y) * sigma_prime(self.z[self.network_depth-2])
        self.dW[self.network_depth-2] += delta.T.dot(self.a[self.network_depth-2])
        self.db[self.network_depth-2] += np.sum(delta.T, axis=1, keepdims=True)

        for l in reversed(range(self.network_depth-2)):
            delta = delta.dot(self.W[l+1]) * relu_prime(self.z[l])
            self.dW[l] += self.a[l].T.dot(delta).T
            self.db[l] += np.sum(delta.T, axis=1, keepdims=True)

        # Gradient descent: Update Weights and Biases
        for l in range(self.network_depth-1):
            self.W[l] += - eta * self.dW[l]
            self.b[l] += - eta * self.db[l]

        # Reset gradients
        self.dW = [np.zeros_like(self.dW[l]) for l in range(self.network_depth-1)]
        self.db = [np.zeros_like(self.db[l]) for l in range(self.network_depth-1)]


    def feedforward(self, X):
        # Feedforward
        self.a[0] = X 
        for l in range(self.network_depth-2):
            self.z[l] = self.a[l].dot(self.W[l].T) + self.b[l].T
            self.a[l+1] = relu(self.z[l])
        self.z[-1] = self.a[-2].dot(self.W[-1].T) + self.b[-1].T
        self.a[-1] = sigma(self.z[-1])
        #return z, a


    def pred(self, X, Y):
        neurons = X
        for l in range(self.network_depth-2):
            neurons = relu(neurons.dot(self.W[l].T) + self.b[l].T)
        logits = neurons.dot(self.W[-1].T) + self.b[-1].T
        accuracy = (np.argmax(logits, axis=1) == np.argmax(Y, axis=1)).sum() / len(X)
        loss = np.sum((Y - sigma(logits))**2) / len(X)
        return loss, accuracy


    def prediction(self, X, Y, epoch, mode):
        loss, accuracy = self.pred(X, Y)
        print('epoch {1} {0}_loss {2:.6f} {0}_accuracy {3:.4f}'.format(mode, epoch, loss, accuracy), flush=True)


    def visualize_weights(self, epoch):
        nrow, ncol = 8, 8
        fig, axes = plt.subplots(nrows = nrow, ncols=ncol, figsize=(ncol, nrow))
        for k, ax in enumerate(axes.flatten()):
            ax.imshow(self.W[0][k].reshape(28,28), cmap="viridis")
            ax.axis("off")
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.02, hspace=0.02)
        plt.savefig('mlp_weights_epoch_{}.png'.format(epoch), dpi=250)
        plt.close()


# Load data
x_train, y_train, x_eval, y_eval, x_test, y_test, n_classes, n_input = data.get_data(dataset="mnist", norm=True, one_hot=True)

# Training parameters
learning_rate = 0.1
batch_size = 32
epochs = 100

# Network configuration
network_config = [n_input, 128, n_classes]

# Initialize network
network = NeuralNetwork(network_config)

# Start training
network.optimize(x_train, y_train, x_eval, y_eval, x_test, y_test, epochs, batch_size, learning_rate)
