import json
import numpy as np

PREDICTION_THRESHOLD = 0.5


def gaussian(x, mu=10, sigma=1.5):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def softmax(x):
    """
    :param x: array of predictions
    :return: array of predictions after softmax activation
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


def relu(x):
    """
    :param x: array of predictions
    :return: array of predictions after relu activation
    """
    return np.maximum(0, x)


def sign(x):
    """
    :param x: array of predictions
    :return: array of predictions after sign activation
    """
    return np.sign(x)


def sigmoid(x):
    """
    :param x: array of predictions
    :return: array of predictions after sigmoid activation
    """
    return 1 / (1 + np.exp(-x))


def leaky_relu(x):
    """
    :param x: array of predictions
    :return: array of predictions after leaky relu activation
    """
    return np.maximum(0.1 * x, x)


class NeuralNetwork:
    def __init__(self, model, weights, biases):
        self.weights = []
        self.biases = []  # New line
        self.activations = []
        for i in range(len(model)):
            self.weights.append(np.array(weights[i]).reshape(model[i][0], model[i][1]))
            self.biases.append(biases[i])  # Update this line
            if model[i][2] == 'relu':
                self.activations.append(relu)
            elif model[i][2] == 'sigmoid':
                self.activations.append(sigmoid)
            elif model[i][2] == 'sign':
                self.activations.append(sign)
            elif model[i][2] == 'leaky_relu':
                self.activations.append(leaky_relu)
            elif model[i][2] == 'softmax':
                self.activations.append(softmax)
            elif model[i][2] == 'gaussian':
                self.activations.append(gaussian)
            else:
                raise Exception(f"Unknown activation function {model[i][2]}")

    def propagate(self, data):
        input_data = data
        z = None
        for i, weights in enumerate(self.weights):
            z = np.dot(input_data, weights) + self.biases[i]  # Updated line
            if self.activations[i] != 0:
                z = self.activations[i](z)
            input_data = z
        return np.ravel(z)


def load_network(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    weights = np.array(data['weights'][0])
    biases = data['biases'][0]
    activation_name = data['activations'][0]
    model = [[weights.shape[0], weights.shape[1], activation_name]]

    return model, [weights], [biases]


def load_test(test_file):
    with open(test_file, 'r') as file:
        lines = file.readlines()
        data = [list(map(int, line.strip())) for line in lines]
    return data


def runnet(network, output_file, X_test):
    # Run the test data through the network
    predictions = []
    for x in X_test:
        z = network.propagate(x)
        # predict using the threshold
        predictions.append(1 if z > PREDICTION_THRESHOLD else 0)

    # Write the predictions to the output file
    with open(output_file, "w") as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")


# Main function
def main():
    saved_data_file = "wnet0.txt"
    testnet0 = "testnet0.txt"
    output_file = "predictions0.txt"

    # initialize the model, weights, activations and biases
    model, weights, biases = load_network(saved_data_file)

    # create the network
    network = NeuralNetwork(model, weights, biases)
    X_test = load_test(testnet0)
    # Run the test data through the network and save predictions to output file
    runnet(network, output_file, X_test)
    print("Predictions saved to output.txt")


if __name__ == '__main__':
    main()
