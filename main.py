import random
import sys
import string
import time
import re
from tqdm import tqdm


def load_common_words():
    with open('dict.txt', 'r') as f:
        common_words = set(f.read().split())
    return sorted(common_words, key=len, reverse=False)


def load_letter_frequencies():
    letter_freq = {}
    with open("Letter_Freq.txt", "r") as letter_freq_file:
        for line in letter_freq_file:
            freq, letter = line.strip().split("\t")
            letter_freq[letter] = float(freq)
    return letter_freq


def load_two_letter_frequencies():
    two_letter_freq = {}
    with open("Letter2_Freq.txt", "r") as two_letter_freq_file:
        for line in two_letter_freq_file:
            freq, letter = line.strip().split("\t")
            two_letter_freq[letter] = float(freq)
    return two_letter_freq


def crossover(parent1, parent2):
    child = {}
    keys = list(parent1.keys())
    crossover_point = random.randint(0, len(keys))
    for i in range(crossover_point):
        child[keys[i]] = parent1[keys[i]]
    for i in range(crossover_point, len(keys)):
        child[keys[i]] = parent2[keys[i]]
    return child


class GeneticAlgorithm:
    def __init__(self, population_size=10, selection_rate=0.3, mutation_rate=0.05, elitism_rate=0.1):
        self.start_time = time.time()
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.encryption_code = self.load_encryption_code()
        self.single_letter_words = set(word for word in self.encryption_code.split() if len(word) == 1)
        if len(self.single_letter_words) > 2:
            print("Error: More than 3 single letter words in encryption code")
            sys.exit(1)
        self.common_words = load_common_words()
        self.letter_freq = load_letter_frequencies()
        self.two_letter_freq = load_two_letter_frequencies()
        self.population = self._create_population()

    def load_encryption_code(self):
        with open('enc.txt', 'r') as f:
            self.single_letter_words = set(word.upper() for word in f.read().split() if len(word) == 1)
            f.seek(0)
            # replace all commas and periods with spaces
            encryption_code = f.read().replace('\n', ' ').upper()
            encryption_code = encryption_code.replace(',', ' ')
            encryption_code = encryption_code.replace('.', ' ')
            encryption_code = encryption_code.replace(';', ' ')
            # remove all single letter words that are not in single_letter_words
            encryption_code = ' '.join(word for word in encryption_code.split() if len(word) > 1 or word in self.single_letter_words)
        return encryption_code

    def _create_population(self):
        population = []
        for _ in range(self.population_size):
            individual = dict(zip(string.ascii_uppercase, random.sample(string.ascii_uppercase, 26)))
            population.append(individual)
        return population

    def get_parents(self):
        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        selection_size = int(self.population_size * self.selection_rate)
        parents = sorted_population[:selection_size]
        return parents

    def _evaluate_fitness(self, individual):
        individual['fitness'] = self._compute_fitness(individual)

    def _mutate(self, individual):
        mutated_individual = dict(individual)
        if random.random() < self.mutation_rate:
            # Get letter keys
            letter_keys = [key for key in mutated_individual.keys() if key != 'fitness']
            # Swap two letter values
            key1, key2 = random.sample(letter_keys, 2)
            mutated_individual[key1], mutated_individual[key2] = mutated_individual[key2], mutated_individual[key1]
        return mutated_individual

    def _evolution_step(self, parents):
        children = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = self._mutate(child)
            children.append(child)
        return children

    def evolve(self):
        # select parents
        parents = self.get_parents()
        # Create new children through crossover
        children = self._evolution_step(parents)
        print("children number: ", len(children))
        for child in children:
            self._evaluate_fitness(child)
        self.population.extend(children)
        # Remove the least fit individuals to maintain population size by first sorting by fitness and then removing
        # the last n individuals where n is the number of children created
        self.population = sorted(self.population, key=lambda x: float(x['fitness']), reverse=True)[
                          :self.population_size]
        self.population = self.population[:-len(children)]


    def _compute_fitness(self, individual):
        fitness = 0

        if individual["A"] in self.single_letter_words or individual["I"] in self.single_letter_words:
            fitness += 5.25
        if individual["A"] in self.single_letter_words and individual["I"] in self.single_letter_words:
            fitness += 5.25

        for word in self.encryption_code.split():
            decrypted_word = ""
            for letter in word:
                decrypted_word += individual[letter]
            if decrypted_word in self.common_words:
                fitness += 1

        letter_freq = {}
        for letter in self.encryption_code:
            if letter in letter_freq:
                letter_freq[letter] += 1
            else:
                letter_freq[letter] = 1

        for letter in letter_freq:
            if letter != ' ':  # Exclude space character
                fitness -= abs(self.letter_freq[letter] * len(self.encryption_code) - letter_freq[letter])

        two_letter_freq = {}
        for i in range(len(self.encryption_code) - 1):
            if self.encryption_code[i] == ' ' or self.encryption_code[i + 1] == ' ':  # Exclude space character
                continue
            pair = self.encryption_code[i] + self.encryption_code[i + 1]
            if pair in two_letter_freq:
                two_letter_freq[pair] += 1
            else:
                two_letter_freq[pair] = 1

        for pair in two_letter_freq:
            fitness -= abs(self.two_letter_freq[pair] * len(self.encryption_code) - two_letter_freq[pair])

        return fitness

    def _compute_fitness_parallel(self, individual):
        return self._compute_fitness(individual)

    def run(self, generations=50):
        # calculate fitness for each individual
        for individual_dict in self.population:
            individual_dict['fitness'] = self._compute_fitness(individual_dict)
        end_time = time.time()
        print("Initial population fitness calculated in {} seconds".format(end_time - self.start_time))
        # evolve the population according to the fitness
        bar = tqdm(total=generations, desc='Generations')
        for generation in range(1, generations + 1):
            self.evolve()
            bar.update(1)
        bar.close()


def main():
    # load encryption code
    algorithm = GeneticAlgorithm(population_size=1000, selection_rate=0.3, mutation_rate=0.05)
    algorithm.run()
    print(algorithm.population[0])


if __name__ == "__main__":
    exit(main())
