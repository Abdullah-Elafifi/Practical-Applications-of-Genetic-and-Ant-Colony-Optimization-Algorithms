import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(55)

def init_pop(pop_size, R_max, R_min):
    return np.random.uniform(R_min, R_max, (pop_size, 2))

def fitness_function(x1, x2):
    return 8 - (x1 + 0.0317) ** 2 + x2 ** 2


def arithmetic_cross(two_parents, Pcross=0.6):
    if np.random.rand() < Pcross:
        alpha = np.random.rand()
        child1 = alpha * two_parents[0] + (1 - alpha) * two_parents[1]
        child2 = alpha * two_parents[1] + (1 - alpha) * two_parents[0]
        return child1, child2
    return two_parents[0], two_parents[1]


def gaussian_mutate(individual, sigma=0.5, pMut=0.05, R_max=2, R_min=-2):
    if np.random.rand() < pMut:
        individual += np.random.normal(0, sigma, size=individual.shape)
        individual = np.clip(individual, R_min, R_max)
    return individual


def tournament_selection(pop, fitness_values, k):
    selected = []
    for _ in range(len(pop)):
        participants = np.random.choice(len(pop), k, replace=False)
        winner = participants[np.argmax(fitness_values[participants])]
        selected.append(pop[winner])
    return np.array(selected)


def genetic_algorithm(pop_size=100, generations=100, Pcross=0.6, pMut=0.05, sigma=0.5, R_min=-2, R_max=2, k=2,
                      elitism=2):
    pop = init_pop(pop_size, R_max, R_min)
    best_fitness_values, avg_fitness_values = [], []

    for gen in range(generations):
        fitness_values = np.array([fitness_function(ind[0], ind[1]) for ind in pop])
        best_fitness_values.append(np.max(fitness_values))
        avg_fitness_values.append(np.mean(fitness_values))

        print(f"Generation {gen + 1}: Best Fitness = {best_fitness_values[-1]}")

        sorted_indices = np.argsort(-fitness_values)
        next_pop = pop[sorted_indices[:elitism]]

        selected_parents = tournament_selection(pop, fitness_values, k)
        children = []
        for i in range(0, len(selected_parents) - 1, 2):
            c1, c2 = arithmetic_cross([selected_parents[i], selected_parents[i + 1]], Pcross)
            children.extend(
                [gaussian_mutate(c1, sigma, pMut, R_max, R_min), gaussian_mutate(c2, sigma, pMut, R_max, R_min)])

        pop = np.vstack((next_pop, np.array(children)[:pop_size - elitism]))

    return best_fitness_values, avg_fitness_values, max(best_fitness_values), np.mean(avg_fitness_values)


k_values = [int(input("Enter k value for tournament selection: ")) for _ in range(2)]

best_fitness_k1, avg_fitness_k1, best_overall_k1, avg_overall_k1 = genetic_algorithm(k=k_values[0], elitism=2)
best_fitness_k2, avg_fitness_k2, best_overall_k2, avg_overall_k2 = genetic_algorithm(k=k_values[1], elitism=2)

best_fitness_no_elitism, avg_fitness_no_elitism, best_overall_no_elitism, avg_overall_no_elitism = genetic_algorithm(
    k=k_values[0], elitism=0)

plt.figure(figsize=(10, 5))
plt.plot(best_fitness_k1, label=f"Best Fitness (k={k_values[0]}, Elitism)")
plt.plot(best_fitness_k2, label=f"Best Fitness (k={k_values[1]}, Elitism)")
plt.plot(best_fitness_no_elitism, label="Best Fitness (No Elitism)", linestyle='--')
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.title("Tournament Selection Impact on Fitness Evolution with and without Elitism")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(avg_fitness_k1, label=f"Average Fitness (k={k_values[0]}, Elitism)")
plt.plot(avg_fitness_k2, label=f"Average Fitness (k={k_values[1]}, Elitism)")
plt.plot(avg_fitness_no_elitism, label="Average Fitness (No Elitism)", linestyle='--')
plt.xlabel("Generations")
plt.ylabel("Average Fitness")
plt.title("Tournament Selection Impact on Average Fitness Evolution with and without Elitism")
plt.legend()
plt.show()

print(f"Results for k={k_values[0]} (with elitism):")
print(f"Best Fitness: {best_overall_k1}")
print(f"Average Fitness: {avg_overall_k1}\n")

print(f"Results for k={k_values[1]} (with elitism):")
print(f"Best Fitness: {best_overall_k2}")
print(f"Average Fitness: {avg_overall_k2}\n")

print("Results without elitism:")
print(f"Best Fitness: {best_overall_no_elitism}")
print(f"Average Fitness: {avg_overall_no_elitism}")