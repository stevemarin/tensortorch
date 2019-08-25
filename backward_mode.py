
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC


class Parameter(object):
    def __init__(self, tensor):
        self.tensor = tensor
        self.gradient = np.zeros_like(tensor)


class Layer(ABC):
    def __init__(self):
        self.parameters = []

    def forward(self, X):
        return X, lambda D: D

    def build_param(self, tensor):
        param = Parameter(tensor)
        self.parameters.append(param)
        return param

    def update(self, optimizer):
        for param in self.parameters:
            optimizer.update(param)


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        tensor = np.random.rand(input_dim, output_dim) * np.sqrt(1 / input_dim)
        self.weights = self.build_param(tensor)
        self.bias = self.build_param(np.zeros(output_dim,))

    def forward(self, X):
        def backward(D):
            self.weights.gradient += X.T @ D
            self.bias.gradient += D.sum(axis=0)
            return D @ self.weights.tensor.T
        return X @ self.weights.tensor + self.bias.tensor, backward


class Attention(Layer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, Q, K, V):
        pass


class BatchNormalize(Layer):
    # github.io "understanding the backward pass through BN layer"
    # also from ctherey.github.io "backpropagation"

    def __init__(self, input_dim, eps=1E-9):
        super().__init__()
        self.gamma = self.build_param(np.ones(input_dim))
        self.beta = self.build_param(np.zeros(input_dim))
        self.eps = eps

    def forward(self, X):
        N = X.shape[0]
        mean = X.mean(axis=0)
        var = X.var(axis=0)
        X_hat = (X - mean) * ((var + self.eps) ** -0.5)
        norm = self.gamma.tensor * X_hat + self.beta.tensor

        def backward(D):
            self.beta.gradient += D.sum(axis=0)
            self.gamma.gradient = np.sum(X_hat * D, axis=0)
            return self.gamma.tensor * (var + self.eps) ** -0.5 / N * \
                   (N * D - self.beta.gradient - (X - mean) / (var + self.eps) * np.sum(D * (X - mean), axis=0))

        return norm, backward


class Relu(Layer):
    def forward(self, X):
        mask = X > 0.
        return X * mask, lambda D: D * mask


class Sigmoid(Layer):
    def forward(self, X):
        S = 1. / (1. + np.exp(-X))

        def backward(D):
            return D * S * (1 - S)

        return S, backward


class Softmax(Layer):
    # from stackoverflow post "derivative of softmax function in python"
    def forward(self, X):
        exps = np.exp(X)
        softmax = np.sum(exps)

        def backward(D):
            s = softmax.reshape(-1, 1)
            return np.diagflat(s) - np.dot(s, s.T)

        return softmax, backward


class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for layer in layers:
            self.parameters.extend(layer.parameters)

    def forward(self, X):
        backprops = []
        Y = X
        for layer in self.layers:
            Y, backprop = layer.forward(Y)
            backprops.append(backprop)

        def backward(D):
            for backprop in reversed(backprops):
                D = backprop(D)
            return D

        return Y, backward


class SGDOptimizer(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, param):
        param.tensor -= self.learning_rate * param.gradient
        param.gradient.fill(0.)


def mse_loss(Y_predict, Y_truth):
    diff = Y_predict - Y_truth
    return np.square(diff).mean(), 2 * diff / len(diff)


class Learner(object):
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def fit_batch(self, X, Y):
        Y_, backward = self.model.forward(X)
        L, D = self.loss(Y_, Y)
        backward(D)
        self.model.update(self.optimizer)
        return L

    def fit(self, X, Y, epochs, batch_size):
        losses = []
        for epoch in range(epochs):
            p = np.random.permutation(len(X))
            X, Y = X[p], Y[p]
            loss = 0.0
            for i in range(0, len(X), batch_size):
                loss += self.fit_batch(X[i:i + batch_size], Y[i:i + batch_size])
            losses.append(loss)
        return losses


if __name__ == '__main__':

    # TODO plot training loss

    epochs = 1000
    num_points = 1000
    training_size = 750

    X = np.random.rand(num_points, 2)
    Y = 6.2 * X[:, 0] * X[:, 1] + 3.4 * X[:, 0] * X[:, 0] + 22.2 * np.cos(X[:, 1]) + 6.2
    Y += np.random.normal(loc=0, scale=0.1, size=Y.shape)
    Y = Y.reshape((num_points, 1))

    X_train, Y_train = X[:training_size], Y[:training_size]
    X_test, Y_test = X[training_size:], Y[training_size:]

    model1 = Sequential(Linear(2, 1))
    model2 = Sequential(
        Linear(2, 2),
        Sigmoid(),
        Linear(2, 2),
        Sigmoid(),
        Linear(2, 1)
    )
    model3 = Sequential(
        Linear(2, 16),
        BatchNormalize(16),
        Sigmoid(),
        Linear(16, 16),
        BatchNormalize(16),
        Sigmoid(),
        Linear(16, 1),
    )

    losses1 = Learner(model1, mse_loss, SGDOptimizer(learning_rate=0.01)
                      ).fit(X_train, Y_train, epochs=epochs, batch_size=50)
    losses2 = Learner(model2, mse_loss, SGDOptimizer(learning_rate=0.01)
                      ).fit(X_train, Y_train, epochs=epochs, batch_size=50)
    losses3 = Learner(model3, mse_loss, SGDOptimizer(learning_rate=0.01)
                      ).fit(X_train, Y_train, epochs=epochs, batch_size=50)

    print(mse_loss(model1.forward(X_train)[0], Y_train)[0], mse_loss(model1.forward(X_test)[0], Y_test)[0])
    print(mse_loss(model2.forward(X_train)[0], Y_train)[0], mse_loss(model2.forward(X_test)[0], Y_test)[0])
    print(mse_loss(model3.forward(X_train)[0], Y_train)[0], mse_loss(model3.forward(X_test)[0], Y_test)[0])

    plt.semilogy(losses1, label='losses1')
    plt.semilogy(losses2, label='losses2')
    plt.semilogy(losses3, label='losses3')
    plt.legend(loc='best')
    plt.show()

