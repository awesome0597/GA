import random

import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import json

# Constants
THRESHOLD = 0.98
PREDICTION_THRESHOLD = 0
LAMARCKIAN = 0.15
LOCAL_MINIMA = 30
POPULATION_SIZE = 150
GENERATIONS = 300
MUTATION_RATE = 0.015
SELECTION_RATE = 0.5


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


def predict(predictions, targets):
    """
    :param predictions: array of predictions
    :param targets: array of targets
    :return: accuracy of predictions
    """
    accuracy = np.mean((predictions > PREDICTION_THRESHOLD).astype(int) == targets)
    return accuracy


def update_convergence_data(population, generations, Iteration, convergence_data):
    """
    :param population: population of individuals
    :param generations: number of generations
    :param Iteration: number of iterations
    :param convergence_data: list of convergence data
    """
    Iteration.append(generations)
    fitness_scores = [individual.fitness for individual in population]
    mean_fitness = np.mean(fitness_scores)
    max_fitness = np.max(fitness_scores)
    convergence_data.append([mean_fitness, max_fitness])


def unflatten(flattened, shapes):
    """
    function to unflatten a flattened array
    :param flattened: array to unflatten
    :param shapes:  shapes of the array
    :return:  unflattened array
    """
    new_array = []
    index = 0
    for shape in shapes:
        size = np.product(shape)
        new_array.append(flattened[index: index + size].reshape(shape))
        index += size
    return new_array


class GeneticAlgorithm:
    """
    Genetic algorithm class
    """

    class Individual:
        """
        Individual class
        """
        X = None  # Static class variable to hold X
        y = None  # Static class variable to hold y

        class NeuralNetwork:
            """
              Neural network class
              """

            def __init__(self, model):
                """
                            :param model: given model
                            """
                self.weights = []  # list of weights
                self.biases = []  # New line
                self.activations = []  # list of activations
                for layer in model:
                    input_size = layer[0]
                    output_size = layer[1]
                    activation = layer[2]
                    layer_weights = np.random.uniform(-1, 1, size=(input_size, output_size))
                    layer_biases = np.random.randn(output_size)
                    self.weights.append(layer_weights)
                    self.biases.append(layer_biases)  # New line
                    # append activations to list
                    if activation == 'relu':
                        self.activations.append(relu)
                    elif activation == 'sign':
                        self.activations.append(sign)
                    elif activation == 'sigmoid':
                        self.activations.append(sigmoid)
                    elif activation == 'leaky_relu':
                        self.activations.append(leaky_relu)
                    else:
                        raise Exception(f"Non-supported activation function: {activation}")

            def propagate(self, data):
                """
                :param data: data to propagate
                :return: predictions
                """
                input_data = data
                a = None
                for i, weights in enumerate(self.weights):
                    z = np.dot(input_data, weights) + self.biases[i]  # Updated line
                    if self.activations[i] != 0:
                        a = self.activations[i](z)
                    input_data = a
                return np.ravel(a)

        def __init__(self, model):
            """
            :param model: given model
            """
            self.neural_network = self.NeuralNetwork(model)
            self.fitness = 0
            self.exit_generation = 0

        def calculate_fitness(self):
            """
            calculate fitness of individual
            """
            predictions = self.neural_network.propagate(GeneticAlgorithm.Individual.X)
            accuracy = np.mean((predictions > PREDICTION_THRESHOLD).astype(int) == GeneticAlgorithm.Individual.y)
            self.fitness = round(float(accuracy), 4)

    def __init__(self, X, y, population_size=POPULATION_SIZE, generations=GENERATIONS, threshold=THRESHOLD,
                 selection_rate=SELECTION_RATE, mutation_rate=MUTATION_RATE):
        """
        :param X: training data
        :param y: labels
        :param population_size: population size
        :param generations: number of maximum generations
        :param threshold: value of fitness to stop at
        :param selection_rate: how many individuals to select
        :param mutation_rate: rate of mutation
        """
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.threshold = threshold
        GeneticAlgorithm.Individual.X = X  # Set X as a static variable in Individual
        GeneticAlgorithm.Individual.y = y  # Set y as a static variable in Individual

    def run(self, model):
        Iteration = []
        convergence_data = []
        generations = 0
        # initialize population
        max_fitness_prev = -float('inf')
        population = [self.Individual(model) for _ in range(self.population_size)]
        for individual in population:
            individual.calculate_fitness()

        # sort population by fitness
        population = sorted(population, key=lambda x: x.fitness, reverse=True)

        #calculate fitness of population
        if population[0].fitness > max_fitness_prev:
            max_fitness_prev = population[0].fitness

        local_minima = 0
        with tqdm(total=self.generations) as pbar:
            while generations < self.generations:
                update_convergence_data(population, generations, Iteration, convergence_data)
                pbar.set_description(
                    f'Generation: {generations} Mean Fitness: {convergence_data[-1][0]:.4f} Max Fitness: '
                    f'{convergence_data[-1][1]:.4f}')

                # select elite individuals
                elite = self.selection(population)
                # crossover elite individuals
                children = self.crossover(elite, model)
                # mutate children
                mutated_children = self.mutation(children)

                # calculate fitness of mutated children
                for individual in mutated_children:
                    individual.calculate_fitness()
                population.extend(mutated_children)
                population = sorted(population, key=lambda x: x.fitness, reverse=True)
                population = population[:self.population_size]
                # lamarkian evolution
                population = self.lamarckian_evolution(population)
                # calculate fitness of population
                max_fitness = population[0].fitness
                if max_fitness <= max_fitness_prev:
                    local_minima += 1
                    if local_minima >= LOCAL_MINIMA:
                        if any(individual.fitness > self.threshold for individual in population):
                            print('Threshold met at generation', generations, '!')
                            population[0].exit_generation = generations
                            break
                        else:
                            print('Local minima reached. Exiting...')
                            population[0].exit_generation = generations
                            break
                else:
                    local_minima = 0
                    max_fitness_prev = max_fitness
                generations += 1
                pbar.update(1)

            best_individual = population[0]

        # Plot convergence data
        convergence_data = np.array(convergence_data)
        plt.plot(Iteration, convergence_data[:, 0], label='Mean Fitness')
        plt.plot(Iteration, convergence_data[:, 1], label='Max Fitness')
        plt.title('Convergence nn1')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.savefig('convergence_nn1.png')

        return best_individual

    def selection(self, population):
        """
         :param population: population of individuals
        """
        sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        parents = sorted_population[:int(self.selection_rate * len(sorted_population))]
        return parents

    def crossover(self, population, network):
        """
        :param population: population of individuals
        :param network: given model
        :return: children
        """
        children = []
        for _ in range((len(population))):
            parent1, parent2 = random.sample(population, 2)
            child = self.Individual(network)

            child_weights = []
            child_biases = []
            for weights1, weights2, biases1, biases2 in zip(parent1.neural_network.weights,
                                                            parent2.neural_network.weights,
                                                            parent1.neural_network.biases,
                                                            parent2.neural_network.biases):
                genes1 = weights1.flatten()
                genes2 = weights2.flatten()

                split = random.randint(0, len(genes1) - 1)
                child_genes = np.concatenate((genes1[:split], genes2[split:]))

                child_weights.append(child_genes.reshape(weights1.shape))

                bias_genes1 = biases1.flatten()
                bias_genes2 = biases2.flatten()

                split = random.randint(0, len(bias_genes1) - 1)
                child_bias_genes = np.concatenate((bias_genes1[:split], bias_genes2[split:]))

                child_biases.append(child_bias_genes.reshape(biases1.shape))

            child.neural_network.weights = child_weights
            child.neural_network.biases = child_biases
            children.append(child)

        return children

    def mutation(self, population, lamarckian=False):
        mutation_rate = self.mutation_rate
        if lamarckian:
            mutation_rate = LAMARCKIAN

        for individual in population:
            for i in range(len(individual.neural_network.weights)):
                layer_weights = individual.neural_network.weights[i]
                layer_biases = individual.neural_network.biases[i]

                # Mutate the weights
                weight_mask = np.random.choice([True, False], size=layer_weights.shape,
                                               p=[mutation_rate, 1 - mutation_rate])
                layer_weights[weight_mask] = np.random.normal(0, 0.1, size=weight_mask.sum())

                # Mutate the biases
                bias_mask = np.random.choice([True, False], size=layer_biases.shape,
                                             p=[mutation_rate, 1 - mutation_rate])
                layer_biases[bias_mask] = np.random.normal(0, 0.1, size=bias_mask.sum())

        return population

    def lamarckian_evolution(self, population):
        for i in range(len(population)):
            individual = population[i]
            new_individual = copy.deepcopy(individual)
            new_individual = self.mutation([new_individual], lamarckian=True)[0]
            new_individual.calculate_fitness()
            if new_individual.fitness > individual.fitness:
                population[i] = new_individual
        return population


def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [list(map(int, line.strip().replace(" ", ""))) for line in lines]
    features = [sample[:-1] for sample in data]
    labels = [sample[-1] for sample in data]
    return np.array(features), np.array(labels)


def save_data(best_individual, model):
    data = {
        'weights': [layer.tolist() for layer in best_individual.neural_network.weights],
        'biases': [layer.tolist() for layer in best_individual.neural_network.biases],
        'activations': [str(layer[2]) for layer in model]
    }
    with open("wnet1.txt", 'w') as file:
        json.dump(data, file)


def main():
    learning_file = "train_file1.txt"
    test_file = "test_file1.txt"
    testnet = "testnet1.txt"
    X_train, y_train = load_data(learning_file)
    X_test, y_test = load_data(test_file)

    network = [[16, 1, "sign"]]
    ga = GeneticAlgorithm(X=X_train, y=y_train)
    best_individual = ga.run(network)
    test_predictions = best_individual.neural_network.propagate(X_test)
    print("Test Set Accuracy:", predict(test_predictions, y_test))

    save_data(best_individual, network)


if __name__ == "__main__":
    main()
