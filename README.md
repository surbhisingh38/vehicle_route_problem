
Vehicle Routing Problem (VRP) using Genetic Algorithm

This project solves the Vehicle Routing Problem (VRP) using a Genetic Algorithm (GA).
It demonstrates how evolutionary techniques can optimize vehicle routes to minimize total travel distance while respecting vehicle capacity constraints.

🚚 Problem Description

The Vehicle Routing Problem (VRP) is a combinatorial optimization problem where:
A depot (starting location) must serve multiple customers.
Each customer has a demand.
A fleet of vehicles, each with a limited capacity, must deliver goods.
The goal is to minimize the total distance traveled while ensuring no vehicle exceeds its capacity.

🧬 Genetic Algorithm Approach

The GA evolves solutions over multiple generations:
Initialization – Generate a random population of solutions (chromosomes).
Fitness Function – Evaluate total route distance + penalty for exceeding vehicle capacity.
Selection – Use tournament selection to pick parents.
Crossover – Apply Edge Recombination Crossover (ERX) to create offspring while preserving adjacency.
Mutation – Apply remove-and-reinsert mutation to introduce diversity.
Elitism – Best solutions survive to the next generation.
Termination – Stop after a fixed number of generations.

📊 Results & Visualization

The script outputs:
Best VRP Solution: Routes assigned to each vehicle with total demand.
Fitness Over Generations: Convergence of GA towards an optimal solution.

Example output:
Left Plot: Best VRP routes from depot (red square) to customers (blue circles).
Right Plot: Fitness (total distance) decreasing over generations.

⚙️ How to Run
Clone this repository or download the project file.
Install dependencies:
pip install matplotlib

Run the script:
python vrp_project.py

🔧 Parameters

You can tune the following hyperparameters inside vrp_project.py:

POPULATION_SIZE = 100
GENERATIONS = 1000
ELITE_SIZE = 10         # number of best individuals preserved
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.10


Problem setup:

NUM_CUSTOMERS = 20
NUM_VEHICLES = 4
VEHICLE_CAPACITY = 50

📌 Example Output (Console)
--- Results ---
Best solution's total distance (fitness): 785.32
Routes:
  Vehicle 1: Depot -> 5 -> 8 -> 2 -> Depot | Demand: 48/50
  Vehicle 2: Depot -> 7 -> 9 -> 1 -> Depot | Demand: 50/50
  Vehicle 3: Depot -> 6 -> 3 -> Depot       | Demand: 32/50
  Vehicle 4: Depot -> 4 -> 10 -> Depot      | Demand: 29/50

📚 Key Learnings

Genetic Algorithms can efficiently solve NP-hard optimization problems like VRP.
Proper use of crossover and mutation helps preserve good structures while exploring new ones.
Visualization helps in analyzing solution quality and convergence speed.
