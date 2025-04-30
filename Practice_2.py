import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(2)

def binary_to_real(binary_string, lower_bound, upper_bound, num_bits):
    decimal_value = int(binary_string, 2)
    return lower_bound + (decimal_value / (2 ** num_bits - 1)) * (upper_bound - lower_bound)

def gray_to_binary(gray_string):
    binary = gray_string[0]
    for i in range(1, len(gray_string)):
        binary += str(int(binary[-1]) ^ int(gray_string[i]))
    return binary

def decode_chromosome(chromosome, num_bits, lower_bound, upper_bound, use_gray=False):
    x1_bits = chromosome[:num_bits]
    x2_bits = chromosome[num_bits:]

    if use_gray:
        x1_bits = gray_to_binary(x1_bits)
        x2_bits = gray_to_binary(x2_bits)

    x1 = binary_to_real(x1_bits, lower_bound, upper_bound, num_bits)
    x2 = binary_to_real(x2_bits, lower_bound, upper_bound, num_bits)

    return x1, x2

def fitness_function(x1, x2, constraint=False):
    fitness = 8 - (x1 + 0.0317) ** 2 + x2 ** 2
    if constraint:
        fitness -= abs(x1 + x2 - 1)
    return fitness

class GeneticAlgorithm:
    def __init__(self, population_size, num_bits, generations, crossover_prob, mutation_prob, lower_bound, upper_bound,
                 use_gray=False, constraint=False, elitism=2):
        self.population_size = population_size
        self.num_bits = num_bits
        self.chromosome_length = 2 * num_bits
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.use_gray = use_gray
        self.constraint = constraint
        self.elitism = elitism

    def initialize_population(self):
        return ["".join(random.choice("01") for _ in range(self.chromosome_length)) for _ in
                range(self.population_size)]

    def evaluate_fitness(self, chromosome):
        x1, x2 = decode_chromosome(chromosome, self.num_bits, self.lower_bound, self.upper_bound, self.use_gray)
        return fitness_function(x1, x2, self.constraint)

    def select_parents(self, population, fitness_scores):
        probabilities = np.array(fitness_scores) / np.sum(fitness_scores) if np.sum(
            fitness_scores) > 0 else np.zeros_like(fitness_scores)
        return list(np.random.choice(population, size=2, p=probabilities))

    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_prob:
            point = np.random.randint(1, self.chromosome_length)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1, parent2

    def mutate(self, chromosome):
        return "".join(str(1 - int(bit)) if np.random.rand() < self.mutation_prob else bit for bit in chromosome)

    def run(self):
        population = self.initialize_population()
        best_fitness_vector, avg_fitness_vector = [], []

        for _ in range(self.generations):
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]
            best_fitness_vector.append(max(fitness_scores))
            avg_fitness_vector.append(np.mean(fitness_scores))

            sorted_population = [pop for _, pop in sorted(zip(fitness_scores, population), reverse=True)]
            next_population = sorted_population[:self.elitism]

            while len(next_population) < self.population_size:
                p1, p2 = self.select_parents(population, fitness_scores)
                c1, c2 = self.crossover(p1, p2)
                next_population.extend([self.mutate(c1), self.mutate(c2)])

            population = next_population[:self.population_size]

        return best_fitness_vector, avg_fitness_vector, max(best_fitness_vector), np.mean(avg_fitness_vector)

num_bits_options = [5,8,12]
encoding_types = [False, True]

plt.figure(figsize=(12, 6))

for num_bits in num_bits_options:
    for use_gray in encoding_types:
        ga = GeneticAlgorithm(population_size=100, num_bits=num_bits, generations=100,
                              crossover_prob=0.6, mutation_prob=0.01, lower_bound=-2, upper_bound=2,
                              use_gray=use_gray)
        best_fitness, avg_fitness, best_overall, avg_overall = ga.run()

        label = f"{'Gray' if use_gray else 'Binary'} - {num_bits} bits"
        plt.plot(best_fitness, label=label)

        print(f"Encoding: {'Gray' if use_gray else 'Binary'}, Bits: {num_bits}")
        print(f"Best Fitness: {best_overall}")
        print(f"Average Fitness: {avg_overall}\n")

plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.title("Comparison of Encoding and Precision in GA")
plt.legend()
plt.show()

ga_elitism = GeneticAlgorithm(population_size=100, num_bits=12, generations=100,
                              crossover_prob=0.6, mutation_prob=0.01, lower_bound=-2, upper_bound=2,
                              use_gray=False, constraint=False, elitism=2)
ga_no_elitism = GeneticAlgorithm(population_size=100, num_bits=12, generations=100,
                                 crossover_prob=0.6, mutation_prob=0.01, lower_bound=-2, upper_bound=2,
                                 use_gray=False, constraint=False, elitism=0)

best_fitness_elitism, avg_fitness_elitism, best_overall_elitism, avg_overall_elitism = ga_elitism.run()
best_fitness_no_elitism, avg_fitness_no_elitism, best_overall_no_elitism, avg_overall_no_elitism = ga_no_elitism.run()

plt.figure(figsize=(10, 5))
plt.plot(best_fitness_elitism, label="Best Fitness (Elitism)", linestyle='-', color='blue')
plt.plot(best_fitness_no_elitism, label="Best Fitness (No Elitism)", linestyle='--', color='red')
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.title("Best Fitness Over Generations With and Without Elitism")
plt.legend()
plt.show()

print("With Elitism - Best Fitness:", best_overall_elitism)
print("With Elitism - Average Fitness:", avg_overall_elitism)
print("Without Elitism - Best Fitness:", best_overall_no_elitism)
print("Without Elitism - Average Fitness:", avg_overall_no_elitism)