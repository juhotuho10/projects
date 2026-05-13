import numpy as np
from prettytable import PrettyTable


def logsig(x):
    return 1 / (1 + np.exp(-x))


def train_test(X, y, network):
    table = PrettyTable()
    table.field_names = ["real", "predicted", "raw value"]
    for _ in range(10000):
        for input, target in zip(X, y):
            network.train(input, target)

    print("real, rounded, raw number")
    for i, input in enumerate(X):
        pred = network.forward(input)
        pred = round(pred, 3)
        table.add_row([y[i], round(pred), pred])
    print(table)
    print()


class perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = logsig(np.dot(self.weights, inputs) + self.bias)
        return self.output

    def update(self, error, inputs):
        dL_dy = error
        dy_din = self.output * (1 - self.output)
        din_dw = inputs
        grad_weights = dL_dy * dy_din * din_dw
        grad_bias = dL_dy * dy_din
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

        return grad_bias


X = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]

X = np.array(X)

# x1 ∧ x2 ∧ x3
y1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])

# x1 ∨ x2 ∨ x3
y2 = np.array([0, 1, 1, 1, 1, 1, 1, 1])

# ((x1 ∧ ¬x2) ∨ (¬x1 ∧ x2)) ∧ x3
y3 = np.array([0, 0, 0, 0, 0, 1, 1, 0])


class Network:
    # network for 3 perceptrons
    def __init__(self, learning_rate=0.001):
        self.hidden_layer = [
            perceptron(input_size=3, learning_rate=learning_rate) for _ in range(2)
        ]
        self.output_neuron = perceptron(input_size=2, learning_rate=learning_rate)

    def forward(self, inputs):
        hidden_outputs = np.array(
            [neuron.forward(inputs) for neuron in self.hidden_layer]
        )
        return self.output_neuron.forward(hidden_outputs)

    def train(self, inputs, target):
        self.forward(inputs)

        output_error = -(target - self.output_neuron.output)

        hidden_outputs = np.array([neuron.output for neuron in self.hidden_layer])
        probagated_error = self.output_neuron.update(output_error, hidden_outputs)

        for i, neuron in enumerate(self.hidden_layer):
            # backprobagated error
            hidden_error = probagated_error * self.output_neuron.weights[i]
            neuron.update(hidden_error, inputs)


network = Network(learning_rate=0.01)

print("Neural Network test for y1")
train_test(X, y1, network)

# ------------------------------------------------------------------------------------------
print("Neural Network test for y2")
train_test(X, y2, network)

# ------------------------------------------------------------------------------------------

print("Neural Network test for y3")
train_test(X, y3, network)

# ------------------------------------------------------------------------------------------
