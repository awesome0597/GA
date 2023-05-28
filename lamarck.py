import random
import string
from tqdm import tqdm
from main import GeneticAlgorithm


class LamarckianGeneticAlgorithm(GeneticAlgorithm):
    def _evaluate_fitness(self, individual):
        self.fitness_counter += 1
        decrypted_text = self.decrypt_individual(individual)
        individual['fitness'] = self._compute_fitness(decrypted_text)
        individual.update(self.extract_decryption_key(decrypted_text))

    def decrypt_individual(self, individual):
        decryption_key = individual.copy()

        # Decrypt the encryption code using the individual's decryption key
        encrypted_code = self.encryption_code.copy()

        for key in decryption_key:
            if key != 'fitness' and key != 'word_percent':
                encrypted_code = encrypted_code.replace(key, decryption_key[key].lower())

        return encrypted_code

    def extract_decryption_key(self, decrypted_text):
        # Extract the decryption key from the decrypted text
        decryption_key = {}

        for i, char in enumerate(string.ascii_uppercase):
            decrypted_char = decrypted_text[i]
            decryption_key[decrypted_char.upper()] = char

        return decryption_key

    def mutate(self, individual):
        mutated_individual = super().mutate(individual)
        return mutated_individual

    def run(self):
        best_individual, decrypted_text = super().run()

        print("Best Decryption Key: " + str(best_individual) + "\n")
        print("Decrypted Text:\n\n" + decrypted_text)

        return best_individual, decrypted_text


# Create a new instance of LamarckianGeneticAlgorithm and run it
lamarckian_ga = LamarckianGeneticAlgorithm(encryption_code, target_text, mutation_rate)
best_individual, decrypted_text = lamarckian_ga.run()
