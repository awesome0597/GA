import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm


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


def calculate_accuracy(predictions, targets):
    predictions_binary = np.round(predictions)
    accuracy = np.mean(predictions_binary == targets)
    return accuracy


def calculate_fitness(individual, X, y):
    predictions = individual.neural_network.propagate(X)
    rounded_predictions = np.round(predictions)  # Round predictions to 0 or 1
    accuracy = np.mean(rounded_predictions == y)  # Calculate accuracy
    individual.fitness = accuracy


class GeneticAlgorithm:
    class Individual:
        X = None  # Static class variable to hold X
        y = None  # Static class variable to hold y

        def __init__(self, model):
            self.neural_network = self.NeuralNetwork(model)
            self.fitness = 0

        @staticmethod
        def calculate_fitness(individual):
            predictions = individual.neural_network.propagate(GeneticAlgorithm.Individual.X)
            rounded_predictions = (predictions > 0.5).astype(int)
            accuracy = np.mean(rounded_predictions == GeneticAlgorithm.Individual.y)
            individual.fitness = round(float(accuracy), 4)

        class NeuralNetwork:
            def __init__(self, model):
                self.weights = []
                self.activations = []
                for layer in model:
                    input_size = layer[0]
                    output_size = layer[1]
                    activation = layer[2]
                    self.weights.append(2 * np.random.random((input_size, output_size)) - 1)
                    self.activations.append(activation)

            def propagate(self, data):
                input_data = data
                for i in range(len(self.weights)):
                    z = np.dot(input_data, self.weights[i])
                    a = leaky_relu(z)
                    input_data = a
                yhat = a
                return yhat

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

        population = [self.Individual(model) for _ in range(self.population_size)]
        for individual in population:
            individual.calculate_fitness(individual)

        with tqdm(total=self.generations) as pbar:
            while generations < self.generations:
                Iteration.append(generations)
                fitness_scores = [individual.fitness for individual in population]
                mean_fitness = np.mean(fitness_scores)
                max_fitness = np.max(fitness_scores)
                convergence_data.append([mean_fitness, max_fitness])
                pbar.set_description(
                    f'Generation: {generations} Mean Fitness: {mean_fitness:.4f} Max Fitness: {max_fitness:.4f}')

                # Lamarckian evolution
                for individual in population:
                    new_individual = copy.deepcopy(individual)
                    new_individual = self.mutation([new_individual], lamarckian=True)[0]
                    new_individual.calculate_fitness(individual)
                    if new_individual.fitness > individual.fitness:
                        individual.neural_network = new_individual.neural_network
                        individual.fitness = new_individual.fitness

                population = self.selection(population)
                population = self.crossover(population, model, self.population_size)
                population = self.mutation(population)

                for individual in population:
                    individual.calculate_fitness(individual)

                if any(individual.fitness > self.threshold for individual in population):
                    print('Threshold met at generation', generations, '!')
                    break

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
            child1 = self.Individual(network)
            dimensions = [a.shape for a in parent1.neural_network.weights]

            genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
            genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])

            split = random.randint(0, len(genes1) - 1)
            child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())

            child1.neural_network.weights = self.unflatten(child1_genes, dimensions)

            children.append(child1)
        population.extend(children)
        return population

    def mutation(self, population, lamarckian=False):
        mutation_count = 5
        if lamarckian:
            mutation_count = 10

        for individual in population:
            for _ in range(mutation_count):
                if random.random() < self.mutation_rate:
                    weights = individual.neural_network.weights
                    new_weights = []
                    for weight in weights:
                        shape = weight.shape
                        flattened = weight.flatten()
                        rand_index = random.randint(0, len(flattened) - 1)
                        flattened[rand_index] = 2 * random.random() - 1  # Generate random value between -1 and 1
                        new_weights.append(flattened.reshape(shape))
                    individual.neural_network.weights = new_weights
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

    network = [[16, 1, sigmoid]]  # 16 input features, 1 output neuron

    ga = GeneticAlgorithm(X=X_train, y=y_train, population_size=50, generations=10, threshold=0.9,
                          selection_rate=0.5, mutation_rate=1)

    best_individual = ga.run(network)

    test_predictions = best_individual.neural_network.propagate(X_test)
    test_accuracy = calculate_accuracy(test_predictions, y_test)
    print("Test Set Predictions:", test_predictions)
    print("Test Set Accuracy:", test_accuracy)


if __name__ == '__main__':
    main()
