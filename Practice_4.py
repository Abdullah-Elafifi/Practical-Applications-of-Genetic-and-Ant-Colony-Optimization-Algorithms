import numpy as np
import matplotlib.pyplot as plt
import random
import math

def load_city_coordinates(file_path):
    coordinates = []
    with open(file_path, 'r') as f:
        for line in f:
            _, x, y = line.split()
            coordinates.append((float(x), float(y)))
    return coordinates

def calc_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def initialize_matrices(distance_matrix):
    count = len(distance_matrix)
    pheromone = np.ones((count, count))
    visibility = 1.0 / (distance_matrix + np.eye(count))
    return pheromone, visibility

def generate_path(pheromone, visibility, alpha, beta, total_cities):
    path = [random.randint(0, total_cities - 1)]
    remaining = set(range(total_cities)) - set(path)

    while remaining:
        current = path[-1]
        chances = []
        for city in remaining:
            p = pheromone[current][city] ** alpha
            h = visibility[current][city] ** beta
            chances.append(p * h)

        normalized = [c / sum(chances) for c in chances]
        next_stop = random.choices(list(remaining), weights=normalized)[0]
        path.append(next_stop)
        remaining.remove(next_stop)

    path.append(path[0])
    return path

def evaluate_path_length(path, dist_matrix):
    return sum(dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

def evaporate_and_reinforce(pheromone, tours, lengths, evaporation, shortest_length):
    size = len(pheromone)
    pheromone *= (1 - evaporation)

    best_idx = np.argmin(lengths)
    top_tour = tours[best_idx]
    top_length = lengths[best_idx]

    for route, length in zip(tours, lengths):
        intensity = 1.0 / length
        for i in range(len(route) - 1):
            pheromone[route[i]][route[i + 1]] += intensity

    boost = 1.0 / top_length
    for i in range(len(top_tour) - 1):
        pheromone[top_tour[i]][top_tour[i + 1]] += boost

    return pheromone

def ant_colony_tsp_solver(cities, dist_matrix, visibility, m, alpha, beta, rho, epochs):
    best_path = []
    min_length = float('inf')
    pheromone, _ = initialize_matrices(dist_matrix)

    for gen in range(epochs):
        all_routes = []
        route_costs = []

        for _ in range(m):
            route = generate_path(pheromone, visibility, alpha, beta, len(cities))
            cost = evaluate_path_length(route, dist_matrix)
            all_routes.append(route)
            route_costs.append(cost)

            if cost < min_length:
                min_length = cost
                best_path = route

        pheromone = evaporate_and_reinforce(pheromone, all_routes, route_costs, rho, min_length)

        print(f"Generation {gen + 1}, Best Tour Length: {min_length:.12f}")

    best_coords = [cities[i] for i in best_path]
    return best_coords, best_path, min_length

def visualize(cities, best_coords):
    plt.figure(figsize=(10, 8))
    x_coords = [x for x, _ in cities]
    y_coords = [y for _, y in cities]

    plt.scatter(x_coords, y_coords, s=60, color='navy')

    route_x = [x for x, _ in best_coords]
    route_y = [y for _, y in best_coords]

    plt.plot(route_x, route_y, color='crimson', linewidth=2)
    for idx, (x, y) in enumerate(cities):
        plt.text(x, y, str(idx), fontsize=9, ha='center', va='center')

    plt.title("Ant Colony Optimization - TSP")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def runner():
    data_file = "TSPDATA.txt"
    city_list = load_city_coordinates(data_file)
    city_count = len(city_list)

    dist_matrix = np.zeros((city_count, city_count))
    for i in range(city_count):
        for j in range(city_count):
            dist_matrix[i][j] = calc_distance(city_list[i], city_list[j])

    visibility = 1.0 / (dist_matrix + np.eye(city_count))

    m = 10
    alpha = 1
    beta = 5
    rho = 0.5
    iterations = 50

    coordinates, path_indices, path_length = ant_colony_tsp_solver(
        city_list, dist_matrix, visibility, m, alpha, beta, rho, iterations
    )

    print("\nBest Tour Found (City Indexes):")
    print(" -> ".join(map(str, path_indices)))
    print(f"\nBest Tour Length: {path_length:.12f}")
    print("\nParameters Used:")
    print(f"Alpha: {alpha}")
    print(f"Beta: {beta}")
    print(f"Rho: {rho}")
    print(f"Ants: {m}")
    print(f"Iterations: {iterations}")

    visualize(city_list, coordinates)

if __name__ == "__main__":
    runner()
