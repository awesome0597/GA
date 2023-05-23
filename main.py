import random
import sys
import string
import re
from tqdm import tqdm


def load_common_words():
    with open('dict.txt', 'r') as f:
        common_words = set(f.read().split())
    return common_words


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


COMMON_WORDS = load_common_words()
LETTER_FREQ = load_letter_frequencies()
TWO_LETTER_FREQ = load_two_letter_frequencies()


def crossover(parent1, parent2):
    child = {}
    keys = list(parent1.keys())
    crossover_point = random.randint(0, len(keys))
    for i in range(crossover_point):
        child[keys[i]] = parent1[keys[i]]
    for i in range(crossover_point, len(keys)):
        child[keys[i]] = parent2[keys[i]]
    # Check for duplicate values
    values = set(child.values())
    while len(values) != len(keys):
        unused_letters = list(set(string.ascii_uppercase) - set(child.values()))
        for key, value in child.items():
            if list(child.values()).count(value) > 1:
                new_value = random.choice(unused_letters)
                child[key] = new_value
                unused_letters.remove(new_value)
        values = set(child.values())
    return child


class GeneticAlgorithm:
    def __init__(self, population_size=10, selection_rate=0.3, mutation_rate=0.05, convergence_generations=10,
                 convergence_threshold=0.001):
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.encryption_code, self.letter_freq, self.two_letter_freq = self.load_encryption_code()
        self.single_letter_words = list(set(word for word in self.encryption_code.split() if len(word) == 1))
        if len(self.single_letter_words) > 2:
            print("Error: More than 3 single letter words in encryption code")
            sys.exit(1)
        self.population = self._create_population()
        self.convergence_generations = convergence_generations
        self.convergence_threshold = convergence_threshold
        self.best_fitness_history = []
        self.fitness_counter = 0
        self.word_percentage = 0

    def load_encryption_code(self):
        with open('enc.txt', 'r') as f:
            self.single_letter_words = set(word.upper() for word in f.read().split() if len(word) == 1)
            f.seek(0)
            # replace all commas and periods with spaces
            encryption_code = f.read().replace('\n', ' ').upper()
            encryption_code = re.sub(r'[.,;]', ' ', encryption_code)
            # remove all single letter words that are not in single_letter_words
            encryption_code = ' '.join(
                word for word in encryption_code.split() if len(word) > 1 or word in self.single_letter_words)
            letter_freq = {}
            # Calculate letter frequencies
            for letter in encryption_code:
                if letter != ' ':
                    if letter in letter_freq:
                        letter_freq[letter] += 1
                    else:
                        letter_freq[letter] = 1
            # Normalize letter frequencies
            total_letters = sum(letter_freq.values())
            for letter in letter_freq:
                letter_freq[letter] /= total_letters

            # Calculate two letter frequencies in the encryption code
            two_letter_freq = {}
            for i in range(len(encryption_code) - 1):
                two_letter = encryption_code[i:i + 2]
                if two_letter in two_letter_freq:
                    two_letter_freq[two_letter] += 1
                else:
                    two_letter_freq[two_letter] = 1
            # Normalize two letter frequencies
            total_two_letters = sum(two_letter_freq.values())
            for two_letter in two_letter_freq:
                two_letter_freq[two_letter] /= total_two_letters
        return encryption_code, letter_freq, two_letter_freq

    def _create_population(self):
        population = []
        abcPercent = int(self.population_size * 0.1)
        singleLetterPercent = int(self.population_size * 0.1)
        rest = self.population_size - abcPercent - singleLetterPercent
        for i in range(abcPercent):
            individual = dict(zip(string.ascii_uppercase, string.ascii_uppercase))
            population.append(individual)

        for i in range(abcPercent, singleLetterPercent):
            # set the letters A and I as values randomly in the individual based on the keys in single_letter_words
            individual = dict(zip(string.ascii_uppercase, random.sample(string.ascii_uppercase, 26)))
            # find where the letters A and I are in the individual
            for key, value in individual.items():
                if i % 2 == 0:
                    if value == 'A':
                        individual[key] = individual[self.single_letter_words[0]]
                        individual[self.single_letter_words[0]] = 'A'
                    elif value == 'I':
                        individual[key] = individual[self.single_letter_words[1]]
                        individual[self.single_letter_words[1]] = 'I'
                    population.append(individual)
                else:
                    if value == 'A':
                        individual[key] = individual[self.single_letter_words[1]]
                        individual[self.single_letter_words[1]] = 'A'
                    elif value == 'I':
                        individual[key] = individual[self.single_letter_words[0]]
                        individual[self.single_letter_words[0]] = 'I'
                    population.append(individual)

        for _ in range(rest, self.population_size):
            individual = dict(zip(string.ascii_uppercase, random.sample(string.ascii_uppercase, 26)))
            population.append(individual)
        return population

    def get_parents(self):
        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        selection_size = int(self.population_size * self.selection_rate)
        parents = sorted_population[:selection_size]
        return parents

    def _evaluate_fitness(self, individual):
        self.fitness_counter += 1
        individual['fitness'] = self._compute_fitness(individual)

    def _mutate(self, individual):
        mutated_individual = dict(individual)
        if random.random() < self.mutation_rate:
            # Get letter keys
            letter_keys = [key for key in mutated_individual.keys() if key != 'fitness' and key != 'word_percent']
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

    def add_children(self, children):
        for child in children:
            self._evaluate_fitness(child)
        self.population.extend(children)
        self.population = sorted(self.population, key=lambda x: float(x['fitness']), reverse=True)[
                          :self.population_size]
        self.population = self.population[:-len(children)]

    def evolve(self):
        # select parents
        parents = self.get_parents()
        # Create new children through crossover
        children = self._evolution_step(parents)
        # Add children to population
        self.add_children(children)

    def nuke_em(self):
        # shuffle the population
        random.shuffle(self.population)
        # kill double the selection amount
        self.population = self.population[:int(self.selection_rate * self.population_size * 0.5)]
        # mutate the survivors
        for individual in self.population:
            for _ in range(3):
                # Get letter keys
                letter_keys = [key for key in individual.keys() if key != 'fitness' and key != 'word_percent']
                # Swap two letter values
                key1, key2 = random.sample(letter_keys, 2)
                individual[key1], individual[key2] = individual[key2], individual[key1]
        # create new children
        children = []
        for i in range(self.population_size - len(self.population)):
            parent1, parent2 = random.sample(self.population, 2)
            child = crossover(parent1, parent2)
            child = self._mutate(child)
            children.append(child)
        # add children to population
        self.population.extend(children)

    def _compute_fitness(self, individual):
        fitness = 0
        encryption_code = self.encryption_code
        word_percent = 0

        single_word_count = sum(individual[word] in {"A", "I"} for word in self.single_letter_words)
        if single_word_count > 0:
            fitness += 100
        if single_word_count == 2:
            fitness += 100

        for word in encryption_code.split():
            decrypted_word = []
            for letter in word:
                decrypted_word.append(individual[letter])
            decrypted_word = ''.join(decrypted_word)
            if decrypted_word.lower() in COMMON_WORDS:
                fitness += 1
                word_percent += 1

        individual['word_percent'] = word_percent / len(encryption_code.split())

        for key in self.two_letter_freq:
            converted_key = individual[key[0]] + individual[key[1]]  # Generate the two-letter combination using individual dict
            fitness -= abs(TWO_LETTER_FREQ[converted_key] - self.two_letter_freq[key])

        for key in self.letter_freq:
            fitness -= abs(LETTER_FREQ[individual[key]] - self.letter_freq[key])

        return fitness

    def run(self):
        # calculate fitness for each individual
        for individual_dict in self.population:
            individual_dict['fitness'] = self._compute_fitness(individual_dict)
        self.population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        # evolve the population according to the fitness
        target_word_percentage = 1  # The fitness value to track progress towards

        initial_word_percentage = self.population[0]['word_percent']
        generations = 0
        local_minima = 0
        word_percentage = self.population[0]['word_percent']
        #  create bar
        bar = tqdm(total=target_word_percentage, initial=initial_word_percentage, desc="Word Percentage", position=0,
                   leave=True)

        while local_minima <= 100 and self.population[0]['word_percent'] < target_word_percentage:
            self.evolve()
            bar.update(self.population[0]['word_percent'] - word_percentage)
            current_word_percentage = self.population[0]['word_percent']
            if word_percentage == current_word_percentage:
                local_minima += 1
            else:
                local_minima = 0
            word_percentage = current_word_percentage
            if local_minima > 100:
                print("Local minima reached: \n" + str(self.population[0]['word_percent']))
                if self.population[0]['word_percent'] > 0.8:
                    break
                else:
                    self.nuke_em()
                    print("Nuked\n")
                    local_minima = 0
            bar.set_description(f"Word Percentage: {self.population[0]['word_percent']}")
            generations += 1
        bar.close()
        print(f"Generations: {generations}")
        self.decrypt()

    def decrypt(self):
        decryption_key = self.population[0]
        encrypted_code = open('enc.txt', 'r').read().upper()

        # Replace letters in the encrypted code with the decrypted letter or keep special characters
        for key in decryption_key:
            if key != 'fitness' and key != 'word_percent':
                encrypted_code = encrypted_code.replace(key, decryption_key[key].lower())

        # Save the decrypted text to plain.txt
        with open('plain.txt', 'w') as plain_file:
            plain_file.write(encrypted_code)

        # Save the key to perm.txt
        with open('perm.txt', 'w') as perm_file:
            for key, value in decryption_key.items():
                if key != 'fitness':
                    perm_file.write(f"{key}: {value}\n")


def main():
    # load encryption code
    algorithm = GeneticAlgorithm(population_size=2000, selection_rate=0.3, mutation_rate=0.05)
    # run algorithm
    algorithm.run()


if __name__ == "__main__":
    exit(main())
