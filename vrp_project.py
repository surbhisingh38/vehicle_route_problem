import random
import math
import copy
import matplotlib.pyplot as plt

# --- Core Classes for VRP and Genetic Algorithm ---

class VRPProblem:
    """
    Encapsulates the Vehicle Routing Problem data.

    Args:
        locations (dict): A dictionary mapping location IDs to (x, y) coordinates.
        demands (dict): A dictionary mapping location IDs to their demand.
        depot_id (int): The ID of the depot location.
        num_vehicles (int): The number of vehicles available.
        capacity (float): The capacity of each vehicle.
    """
    def __init__(self, locations, demands, depot_id, num_vehicles, capacity):
        self.locations = locations
        self.demands = demands
        self.depot_id = depot_id
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        # Pre-compute the distance matrix for efficiency
        self.distance_matrix = self._compute_distance_matrix()
        # Customer IDs are all locations except the depot
        self.customer_ids = [loc_id for loc_id in locations if loc_id != depot_id]

    def _compute_distance_matrix(self):
        """Creates a matrix of Euclidean distances between all locations."""
        loc_ids = list(self.locations.keys())
        matrix = {}
        for from_id in loc_ids:
            matrix[from_id] = {}
            for to_id in loc_ids:
                if from_id == to_id:
                    matrix[from_id][to_id] = 0
                else:
                    x1, y1 = self.locations[from_id]
                    x2, y2 = self.locations[to_id]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    matrix[from_id][to_id] = dist
        return matrix

    def get_distance(self, from_id, to_id):
        """Returns the pre-computed distance between two locations."""
        return self.distance_matrix[from_id][to_id]

class GeneticAlgorithm:
    """
    The main Genetic Algorithm class for solving the VRP.

    It orchestrates the evolution process, including initialization, selection,
    crossover, and mutation, based on the principles outlined in the paper.
    """
    def __init__(self, problem, pop_size, generations, elite_size, crossover_rate, mutation_rate, tournament_size=5):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        # A high penalty factor ensures that invalid solutions are heavily penalized
        self.OVERLOAD_PENALTY_FACTOR = 1000  

    def _initialize_population(self):
        """
        Creates the initial population of random solutions (chromosomes).
        A chromosome is a permutation of customer IDs and vehicle separators.
        """
        population = []
        # Vehicle separators are represented by IDs greater than the max customer ID
        num_customers = len(self.problem.customer_ids)
        separator_start_id = num_customers + 1
        separators = list(range(separator_start_id, separator_start_id + self.problem.num_vehicles - 1))
        
        base_chromosome = self.problem.customer_ids + separators
        
        for _ in range(self.pop_size):
            chromosome = random.sample(base_chromosome, len(base_chromosome))
            population.append(chromosome)
        return population

    def _parse_chromosome(self, chromosome):
        """
        Splits a chromosome into individual routes for each vehicle.
        Routes are determined by the position of the separator genes.
        """
        routes = []
        current_route = []
        num_customers = len(self.problem.customer_ids)
        
        for gene in chromosome:
            if gene > num_customers:  # It's a separator
                if current_route: # Avoid adding empty routes
                    routes.append(current_route)
                current_route = []
            else:  # It's a customer
                current_route.append(gene)
        
        if current_route: # Add the last route
            routes.append(current_route)
            
        return routes

    def _calculate_fitness(self, chromosome):
        """
        Calculates the fitness of a chromosome. Fitness is the total distance
        of all routes, plus a penalty for any routes that exceed vehicle capacity.
        The goal is to MINIMIZE this value.
        """
        routes = self._parse_chromosome(chromosome)
        total_distance = 0
        total_capacity_overload = 0

        for route in routes:
            if not route:
                continue
            
            # Calculate route distance
            route_distance = self.problem.get_distance(self.problem.depot_id, route[0]) # Depot to first customer
            for i in range(len(route) - 1):
                total_distance += self.problem.get_distance(route[i], route[i+1])
            route_distance += self.problem.get_distance(route[-1], self.problem.depot_id) # Last customer to depot
            total_distance += route_distance
            
            # Calculate route demand and penalty for overload
            route_demand = sum(self.problem.demands[customer_id] for customer_id in route)
            if route_demand > self.problem.capacity:
                total_capacity_overload += (route_demand - self.problem.capacity)
        
        # The final fitness is the distance plus a large penalty for any capacity violation
        fitness = total_distance + total_capacity_overload * self.OVERLOAD_PENALTY_FACTOR
        return fitness

    def _tournament_selection(self, population_with_fitness):
        """
        Selects a parent using k-tournament selection.
        """
        tournament = random.sample(population_with_fitness, self.tournament_size)
        # The winner is the one with the lowest fitness value (minimum distance)
        winner = min(tournament, key=lambda item: item[1])
        return winner[0] # Return the chromosome

    def _edge_recombination_crossover(self, parent1, parent2):
        """
        Performs Edge Recombination Crossover (ERX), a method well-suited
        for permutation-based problems like VRP as it preserves adjacency
        information from both parents.
        """
        # 1. Build Adjacency Map
        adjacency_map = {}
        all_genes = set(parent1)
        for gene in all_genes:
            adjacency_map[gene] = set()

        def add_neighbors(parent):
            for i in range(len(parent)):
                left = parent[i-1]
                right = parent[(i+1) % len(parent)]
                adjacency_map[parent[i]].add(left)
                adjacency_map[parent[i]].add(right)

        add_neighbors(parent1)
        add_neighbors(parent2)

        # 2. Build Offspring
        offspring = []
        current_gene = random.choice(parent1)
        
        while len(offspring) < len(parent1):
            offspring.append(current_gene)
            
            # Remove current_gene from all neighbor lists
            for gene in adjacency_map:
                adjacency_map[gene].discard(current_gene)

            if not adjacency_map[current_gene]:
                # If no neighbors, pick a random unvisited gene
                remaining_genes = all_genes - set(offspring)
                if not remaining_genes:
                    break
                current_gene = random.choice(list(remaining_genes))
            else:
                # Choose the neighbor with the smallest number of its own neighbors
                min_neighbors = float('inf')
                next_gene = -1
                for neighbor in adjacency_map[current_gene]:
                    if len(adjacency_map[neighbor]) < min_neighbors:
                        min_neighbors = len(adjacency_map[neighbor])
                        next_gene = neighbor
                    elif len(adjacency_map[neighbor]) == min_neighbors:
                        # Tie-break randomly
                        if random.random() < 0.5:
                            next_gene = neighbor
                current_gene = next_gene

        return offspring

    def _reinsert_mutation(self, chromosome):
        """
        Performs "remove and reinsert" mutation, as described in the paper.
        A random customer is moved to a new random position.
        """
        # We only move customers, not separators, to preserve the number of routes
        customer_genes_with_indices = [
            (i, gene) for i, gene in enumerate(chromosome) 
            if gene in self.problem.customer_ids
        ]
        
        if not customer_genes_with_indices:
            return chromosome # No customers to mutate

        # Pick a random customer to move
        idx_to_move, gene_to_move = random.choice(customer_genes_with_indices)
        
        # Remove it from the chromosome
        mutant = copy.copy(chromosome)
        mutant.pop(idx_to_move)

        # Pick a random new position and insert it
        new_position = random.randint(0, len(mutant))
        mutant.insert(new_position, gene_to_move)
        
        return mutant

    def run(self):
        """
        Executes the genetic algorithm evolution process.
        """
        population = self._initialize_population()
        best_solution = None
        best_fitness = float('inf')
        history = []

        print("Starting Genetic Algorithm...")
        for generation in range(self.generations):
            # Calculate fitness for the entire population
            pop_with_fitness = [(chrom, self._calculate_fitness(chrom)) for chrom in population]
            
            # Find the best solution in the current generation
            current_best_chrom, current_best_fitness = min(pop_with_fitness, key=lambda item: item[1])
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_chrom
            
            history.append(best_fitness)

            if (generation + 1) % 100 == 0:
                print(f"Generation {generation + 1}/{self.generations} | Best Fitness: {best_fitness:.2f}")

            # Create the next generation
            next_generation = []
            
            # 1. Elitism: Carry over the best individuals
            sorted_pop = sorted(pop_with_fitness, key=lambda item: item[1])
            elites = [chrom for chrom, fit in sorted_pop[:self.elite_size]]
            next_generation.extend(elites)

            # 2. Crossover and Mutation
            while len(next_generation) < self.pop_size:
                parent1 = self._tournament_selection(pop_with_fitness)
                parent2 = self._tournament_selection(pop_with_fitness)
                
                if random.random() < self.crossover_rate:
                    child = self._edge_recombination_crossover(parent1, parent2)
                else:
                    child = copy.copy(parent1) # No crossover, clone parent

                if random.random() < self.mutation_rate:
                    child = self._reinsert_mutation(child)
                
                next_generation.append(child)
            
            population = next_generation
            
        print("Genetic Algorithm finished.")
        return best_solution, best_fitness, history

# --- Utility Function for Visualization ---

def plot_solution_and_history(problem, solution, history):
    """
    Visualizes the best VRP solution and the fitness history over generations.
    """
    routes = GeneticAlgorithm(problem, 0, 0, 0, 0, 0)._parse_chromosome(solution)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: VRP Solution
    ax1.set_title("Best VRP Solution")
    ax1.scatter(
        [problem.locations[loc_id][0] for loc_id in problem.customer_ids],
        [problem.locations[loc_id][1] for loc_id in problem.customer_ids],
        c='blue', label='Customers'
    )
    ax1.scatter(
        problem.locations[problem.depot_id][0],
        problem.locations[problem.depot_id][1],
        c='red', s=100, marker='s', label='Depot'
    )

    num_routes = len(routes)
    # Generate a unique color for each route from the 'jet' colormap
    colors = [plt.cm.jet(i / num_routes) for i in range(num_routes)] if num_routes > 0 else []
    for i, route in enumerate(routes):
        if not route: continue
        route_path = [problem.depot_id] + route + [problem.depot_id]
        ax1.plot(
            [problem.locations[loc_id][0] for loc_id in route_path],
            [problem.locations[loc_id][1] for loc_id in route_path],
            color=colors[i],
            marker='o'
        )

    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Fitness History
    ax2.set_title("Fitness (Total Distance) Over Generations")
    ax2.plot(history)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness (Lower is Better)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Define the VRP Instance (a small, random problem for demonstration)
    NUM_CUSTOMERS = 20
    NUM_VEHICLES = 4
    VEHICLE_CAPACITY = 50
    DEPOT_ID = 0

    # Create random locations and demands
    locations = {DEPOT_ID: (50, 50)}
    demands = {DEPOT_ID: 0}
    for i in range(1, NUM_CUSTOMERS + 1):
        locations[i] = (random.randint(0, 100), random.randint(0, 100))
        demands[i] = random.randint(5, 15)

    # 2. Create the problem instance
    vrp_problem = VRPProblem(
        locations=locations,
        demands=demands,
        depot_id=DEPOT_ID,
        num_vehicles=NUM_VEHICLES,
        capacity=VEHICLE_CAPACITY
    )

    # 3. Set Genetic Algorithm Hyperparameters
    # These values are chosen based on common practice and the paper's experiments.
    # They may require tuning for different problem instances.
    POPULATION_SIZE = 100
    GENERATIONS = 1000
    ELITE_SIZE = 10 # 10% of population
    CROSSOVER_RATE = 0.85
    MUTATION_RATE = 0.10

    # 4. Initialize and run the Genetic Algorithm
    ga = GeneticAlgorithm(
        problem=vrp_problem,
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_size=ELITE_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE
    )
    
    best_solution_chromosome, best_solution_fitness, fitness_history = ga.run()

    # 5. Print and visualize the results
    print("\n--- Results ---")
    print(f"Best solution's total distance (fitness): {best_solution_fitness:.2f}")
    
    final_routes = ga._parse_chromosome(best_solution_chromosome)
    print("Routes:")
    for i, route in enumerate(final_routes):
        route_demand = sum(vrp_problem.demands[cust] for cust in route)
        print(f"  Vehicle {i+1}: Depot -> {' -> '.join(map(str, route))} -> Depot | Demand: {route_demand}/{vrp_problem.capacity}")

    plot_solution_and_history(vrp_problem, best_solution_chromosome, fitness_history)

