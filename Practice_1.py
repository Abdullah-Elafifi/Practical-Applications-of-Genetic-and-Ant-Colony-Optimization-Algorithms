import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(1)

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, generations, crossover_prob, mutation_prob, elitism=2):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism

    def initialize_population(self):
        return np.random.randint(0, 2, (self.population_size, self.chromosome_length))

    def evaluate_fitness(self, individual):
        return np.sum(individual)

    def compute_population_fitness(self, population):
        return np.array([self.evaluate_fitness(ind) for ind in population])

    def select_parents(self, population, fitness_scores):
        probabilities = fitness_scores / np.sum(fitness_scores) if np.sum(fitness_scores) > 0 else np.zeros_like(
            fitness_scores)
        return population[np.random.choice(len(population), size=2, p=probabilities)]

    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_prob:
            point = np.random.randint(1, len(parent1))
            return np.concatenate((parent1[:point], parent2[point:])), np.concatenate(
                (parent2[:point], parent1[point:]))
        return parent1, parent2

    def mutate(self, individual):
        mutation_mask = np.random.rand(len(individual)) < self.mutation_prob
        individual[mutation_mask] = 1 - individual[mutation_mask]
        return individual

    def run(self):
        population = self.initialize_population()

        best_fitness_vector = []
        avg_fitness_vector = []

        for _ in range(self.generations):
            fitness_scores = self.compute_population_fitness(population)

            best_fitness_vector.append(np.max(fitness_scores))
            avg_fitness_vector.append(np.mean(fitness_scores))

            sorted_indices = np.argsort(fitness_scores)[-self.elitism:]
            next_population = population[sorted_indices].tolist()

            while len(next_population) < self.population_size:
                p1, p2 = self.select_parents(population, fitness_scores)
                c1, c2 = self.crossover(p1, p2)
                next_population.extend([self.mutate(c1), self.mutate(c2)])

            population = np.array(next_population[:self.population_size])

        return population, best_fitness_vector, avg_fitness_vector


population_size = 20
chromosome_length = 5
generations = 100
crossover_prob = 0.6
mutation_prob = 0.05

ga = GeneticAlgorithm(population_size, chromosome_length, generations, crossover_prob, mutation_prob)
final_population, best_fitness_vector, avg_fitness_vector = ga.run()

print("Final Population:\n", final_population)
print("Best Fitness per Generation:\n", best_fitness_vector)
print("Average Fitness per Generation:\n", avg_fitness_vector)


def run_ga_with_elitism(elitism):
    ga = GeneticAlgorithm(population_size, chromosome_length, generations, crossover_prob, mutation_prob, elitism)
    _, best_fitness, avg_fitness = ga.run()
    return best_fitness, avg_fitness

best_fitness_elitism, avg_fitness_elitism = run_ga_with_elitism(elitism=2)

best_fitness_no_elitism, avg_fitness_no_elitism = run_ga_with_elitism(elitism=0)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(best_fitness_elitism, label="Best Fitness (Elitism)", linestyle='-', color='blue')
plt.plot(best_fitness_no_elitism, label="Best Fitness (No Elitism)", linestyle='--', color='red')
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.title("Best Fitness Over Generations")
plt.legend()

plt.subplot(a1, 2, 2)
plt.plot(avg_fitness_elitism, label="Avg Fitness (Elitism)", linestyle='-', color='blue')
plt.plot(avg_fitness_no_elitism, label="Avg Fitness (No Elitism)", linestyle='--', color='red')
plt.xlabel("Generations")
plt.ylabel("Average Fitness")
plt.title("Average Fitness Over Generations")
plt.legend()
plt.tight_layout()
plt.show()
