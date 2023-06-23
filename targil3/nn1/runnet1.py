import json
import sys

import numpy as np


def relu(x):
    return np.maximum(0, x)


def sign(x):
    return np.sign(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def leaky_relu(x):
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
        #append the prediction to the list, if the prediction is negative, append 0, else append 1
        if z < 0:
            predictions.append(0)
        else:
            predictions.append(int(z))

    # Write the predictions to the output file
    with open(output_file, "w") as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")


# Main function
def main():
    #recieve argument entered through the command line
    testnet = sys.argv[1]
    #recieve the name of the file from the user
    unlabeled_data_file = input("Enter the name of the unlabeled data file: ")

    saved_data_file = "wnet1.txt"
    output_file = "predictions1.txt"

    # initialize the model, weights, activations and biases
    model, weights, biases = load_network(saved_data_file)

    # create the network
    network = NeuralNetwork(model, weights, biases)
    X_test = load_test(unlabeled_data_file)
    # Run the test data through the network and save predictions to output file
    runnet(network, output_file, X_test)
    print("Predictions saved to predictions1.txt")


if __name__ == '__main__':
    main()
