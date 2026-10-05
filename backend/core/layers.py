import numpy as np


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)
        pass


class Relu:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Loss_CCE:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_conf = y_pred_clip[range(samples), y_true]
        negative_log = -np.log(correct_conf)

        return np.mean(negative_log)


class Activation_Softmax_Loss_CCE:
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss_CCE()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs

        return self.loss.forward(self.outputs, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


class Softmax:
    def __init__(self):
        self.outputs = None

    def forward(self, inputs):
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_val = np.exp(shifted_inputs)

        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.outputs = probabilities
        return probabilities
