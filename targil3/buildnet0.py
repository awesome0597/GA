import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import time


def compute_accuracy_score(y_train, predictions):
    """
        Calculates the accuracy score of the predictions made by the network.
        If a prediction matches its true label, it increments the count of correct predictions.
        The accuracy score is the correct predictions divided by the total predictions.
    """
    num_samples = len(y_train)
    correct_predictions = 0

    # If the prediction is correct, increment the count of correct predictions
    for true_label, predicted_label in zip(y_train, predictions):
        if true_label == predicted_label:
            correct_predictions += 1

    # Compute accuracy as the ratio of correct predictions to total number of samples
    accuracy = correct_predictions / num_samples
    return accuracy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sign(x):
    return np.where(x >= 0, 1, 0)


def leaky_relu(x):
    return np.maximum(0.1 * x, x)


def binary_cross_entropy(predictions, targets):
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1 - epsilon)  # Clip predictions to avoid numerical instability
    loss = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return loss


def predict(predictions, targets):
    predictions_binary = np.round(predictions)
    accuracy = compute_accuracy_score(targets, predictions_binary)
    return accuracy


def update_convergence_data(population, generations, Iteration, convergence_data):
    Iteration.append(generations)
    fitness_scores = [individual.fitness for individual in population]
    mean_fitness = np.mean(fitness_scores)
    max_fitness = np.max(fitness_scores)
    convergence_data.append([mean_fitness, max_fitness])


class GeneticAlgorithm:
    class Individual:
        X = None  # Static class variable to hold X
        y = None  # Static class variable to hold y

        class NeuralNetwork:
            def __init__(self, model):
                self.weights = []
                self.activations = []
                for layer in model:
                    input_size = layer[0]
                    output_size = layer[1]
                    activation = layer[2]
                    layer_weights = 2 * np.random.random((input_size, output_size)) - 1
                    self.weights.append(layer_weights)
                    self.activations.append(activation)

            def propagate(self, data):
                input_data = data
                for i in range(len(self.weights)):
                    z = np.dot(input_data, self.weights[i])
                    if self.activations[i] != 0:
                        z = self.activations[i](z)
                    input_data = z
                yhat = z
                return yhat

        def __init__(self, model):
            self.neural_network = self.NeuralNetwork(model)
            self.fitness = 0

        def calculate_fitness(self):
            predictions = self.neural_network.propagate(GeneticAlgorithm.Individual.X)
            rounded_predictions = (predictions > 0.5).astype(int)
            accuracy = compute_accuracy_score(GeneticAlgorithm.Individual.y, rounded_predictions)
            self.fitness = round(float(accuracy), 4)

    def __init__(self, X, y, population_size=100, generations=100, threshold=0.001, selection_rate=0.2,
                 mutation_rate=0.01, batch_size=1):
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.threshold = threshold
        GeneticAlgorithm.Individual.X = X  # Set X as a static variable in Individual
        GeneticAlgorithm.Individual.y = y  # Set y as a static variable in Individual
        self.batch_size = batch_size

    def run(self, model):
        Iteration = []
        convergence_data = []
        generations = 0
        max_fitness_prev = -float('inf')  # Previous maximum fitness
        population = [self.Individual(model) for _ in range(self.population_size)]
        start_time = time.time()
        for individual in population:
            individual.calculate_fitness()
        end_time = time.time()
        print('Time to calculate fitness for initial population:', end_time - start_time, 'seconds')
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        if population[0].fitness > max_fitness_prev:
            max_fitness_prev = population[0].fitness

        local_minima = 0
        with tqdm(total=self.generations) as pbar:
            while generations < self.generations:
                update_convergence_data(population, generations, Iteration, convergence_data)
                pbar.set_description(
                    f'Generation: {generations} Mean Fitness: {convergence_data[-1][0]:.4f} Max Fitness: {convergence_data[-1][1]:.4f}')

                for individual in population:
                    new_individual = copy.deepcopy(individual)
                    new_individual = self.mutation([new_individual], lamarckian=True)[0]
                    new_individual.calculate_fitness()
                    if new_individual.fitness > individual.fitness:
                        individual.neural_network = new_individual.neural_network
                        individual.fitness = new_individual.fitness

                population = self.selection(population)
                population = self.crossover(population, model, self.population_size)
                population = self.mutation(population)

                for individual in population:
                    individual.calculate_fitness()
                population = sorted(population, key=lambda x: x.fitness, reverse=True)

                if any(individual.fitness > self.threshold for individual in population):
                    print('Threshold met at generation', generations, '!')
                    break

                max_fitness = population[0].fitness
                if max_fitness <= max_fitness_prev:
                    local_minima += 1
                    if local_minima >= 10:
                        print('Local minima reached. Exiting...')
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
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

        return best_individual

    def selection(self, population):
        sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        parents = sorted_population[:int(self.selection_rate * len(sorted_population))]
        return parents

    @staticmethod
    def unflatten(flattened, shapes):
        new_array = []
        index = 0
        for shape in shapes:
            size = np.product(shape)
            new_array.append(flattened[index: index + size].reshape(shape))
            index += size
        return new_array

    def crossover(self, population, network, pop_size):
        children = []
        for _ in range((pop_size - len(population))):
            parent1, parent2 = random.sample(population, 2)
            child = self.Individual(network)

            child_weights = []
            for weights1, weights2 in zip(parent1.neural_network.weights, parent2.neural_network.weights):
                genes1 = weights1.flatten()
                genes2 = weights2.flatten()

                split = random.randint(0, len(genes1) - 1)
                child_genes = np.concatenate((genes1[:split], genes2[split:]))

                child_weights.append(child_genes.reshape(weights1.shape))

            child.neural_network.weights = child_weights
            children.append(child)

        population.extend(children)
        return population

    def mutation(self, population, lamarckian=False):
        mutation_count = 5
        if lamarckian:
            mutation_count = 10

        for individual in population:
            for _ in range(mutation_count):
                mask = np.random.choice([True, False], size=individual.neural_network.weights[0].shape,
                                        p=[self.mutation_rate, 1 - self.mutation_rate])
                individual.neural_network.weights[0][mask] += np.random.normal(0, 0.1, size=mask.sum())

        return population


def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [list(map(int, line.strip())) for line in lines]
    features = [sample[:-1] for sample in data]
    labels = [sample[-1] for sample in data]
    return np.array(features), np.array(labels)


def main():
    learning_file = "learning_file.txt"
    test_file = "test_file.txt"

    X_train, y_train = load_data(learning_file)
    X_test, y_test = load_data(test_file)

    network = [[16, 64, 0], [64, 32, 0], [32, 32, 0], [32, 1, sigmoid]]  # 16 input features, 1 output neuron

    ga = GeneticAlgorithm(X=X_train, y=y_train, population_size=250, generations=20, threshold=0.9,
                          selection_rate=0.6, mutation_rate=0.4)

    best_individual = ga.run(network)
    test_predictions = best_individual.neural_network.propagate(X_test)
    test_accuracy = predict(test_predictions, y_test)
    print("Test Set Predictions:", test_predictions)
    print("Test Set Accuracy:", test_accuracy)


if __name__ == '__main__':
    main()
