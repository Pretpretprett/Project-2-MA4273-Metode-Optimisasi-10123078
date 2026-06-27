import numpy as np
import math
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. DTLZ1 Objective Function
# ------------------------------------------------------------------------------
def dtlz1(x, num_objectives=3):
    """
    DTLZ1 objective function.
    x: 1D array of decision variables (in [0, 1]).
    num_objectives: Number of objectives (M).
    The number of decision variables is n = num_objectives + k - 1.
    Typically for DTLZ1, k = 5.
    """
    n = len(x)
    k = n - num_objectives + 1
    
    # Calculate g
    g = 0.0
    for i in range(n - k, n):
        g += (x[i] - 0.5)**2 - np.cos(20 * np.pi * (x[i] - 0.5))
    g = 100 * (k + g)
    
    # Calculate objectives
    f = np.zeros(num_objectives)
    for i in range(num_objectives):
        val = 0.5 * (1 + g)
        for j in range(num_objectives - i - 1):
            val *= x[j]
        if i > 0:
            val *= (1 - x[num_objectives - i - 1])
        f[i] = val
        
    return f

# ------------------------------------------------------------------------------
# 2. Non-Dominated Sorting (from NSGA-II)
# ------------------------------------------------------------------------------
def fast_non_dominated_sort(objectives):
    """
    Sorts a population into different non-domination levels.
    """
    pop_size = objectives.shape[0]
    S = [[] for _ in range(pop_size)]
    front = [[]]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size, dtype=int)

    for p in range(pop_size):
        S[p] = []
        n[p] = 0
        for q in range(pop_size):
            # Check if p dominates q
            if (objectives[p] <= objectives[q]).all() and (objectives[p] < objectives[q]).any():
                S[p].append(q)
            # Check if q dominates p
            elif (objectives[q] <= objectives[p]).all() and (objectives[q] < objectives[p]).any():
                n[p] += 1
        
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

    i = 0
    while len(front[i]) > 0:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front.append(Q)

    del front[-1]
    return front

# ------------------------------------------------------------------------------
# 3. Crowding Distance Assignment (from NSGA-II)
# ------------------------------------------------------------------------------
def calculate_crowding_distance(objectives, front):
    """
    Calculates crowding distance for a set of solutions in a specific front.
    """
    num_solutions = len(front)
    distances = np.zeros(num_solutions)
    
    if num_solutions == 0:
        return distances
    if num_solutions <= 2:
        return np.full(num_solutions, np.inf)

    num_objectives = objectives.shape[1]
    
    for m in range(num_objectives):
        # Sort front based on objective m
        sorted_indices = np.argsort(objectives[front, m])
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        f_max = objectives[front[sorted_indices[-1]], m]
        f_min = objectives[front[sorted_indices[0]], m]
        
        if f_max - f_min == 0:
            continue
            
        for i in range(1, num_solutions - 1):
            distances[sorted_indices[i]] += (objectives[front[sorted_indices[i+1]], m] - objectives[front[sorted_indices[i-1]], m]) / (f_max - f_min)
            
    return distances

# ------------------------------------------------------------------------------
# 4. Cuckoo Search Operators
# ------------------------------------------------------------------------------
def get_levy_flight_step(beta, size):
    """
    Generates a step size following Lévy distribution using Mantegna's algorithm.
    """
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
               (math.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))))**(1 / beta)
    sigma_v = 1.0

    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)
    
    step = u / (np.abs(v)**(1 / beta))
    return step

def polynomial_mutation(population, pm_prob=None, eta_m=20.0):
    """
    Applies Polynomial Mutation (PM) to the population to help escape local optima
    and ensure perfectly distributed solutions on the true Pareto front.
    """
    pop_size, num_vars = population.shape
    if pm_prob is None:
        pm_prob = 1.0 / num_vars
        
    mutated = np.copy(population)
    for i in range(pop_size):
        for j in range(num_vars):
            if np.random.rand() < pm_prob:
                u = np.random.rand()
                if u <= 0.5:
                    delta_q = (2*u)**(1.0/(eta_m+1)) - 1.0
                    mutated[i, j] = mutated[i, j] + delta_q * mutated[i, j]
                else:
                    delta_q = 1.0 - (2.0*(1.0-u))**(1.0/(eta_m+1))
                    mutated[i, j] = mutated[i, j] + delta_q * (1.0 - mutated[i, j])
    return np.clip(mutated, 0.0, 1.0)

def get_cuckoos(population, best_solutions, beta=1.5, alpha=0.1):
    """
    Generate new solutions using global random walk (Lévy flights).
    """
    pop_size, num_vars = population.shape
    new_population = np.zeros_like(population)
    
    for i in range(pop_size):
        # Randomly pick a best solution to fly towards
        best = best_solutions[np.random.randint(0, len(best_solutions))]
        step = get_levy_flight_step(beta, num_vars)
        
        # Global random walk
        new_population[i] = population[i] + alpha * step * (population[i] - best)
        
        # Enforce bounds [0, 1]
        new_population[i] = np.clip(new_population[i], 0.0, 1.0)
        
    # Apply polynomial mutation to ensure fine convergence and diversity
    new_population = polynomial_mutation(new_population)
    return new_population

def empty_nests(population, pa=0.25):
    """
    Discover a fraction pa of nests and build new ones (local random walk).
    """
    pop_size, num_vars = population.shape
    new_population = np.copy(population)
    
    # Probability matrix
    K = np.random.rand(pop_size, num_vars) < pa
    
    perm1 = np.random.permutation(pop_size)
    perm2 = np.random.permutation(pop_size)
    
    # Differential Evolution-style crossover step
    # We use F=0.5 instead of a random tiny step size to perform proper exploration
    new_population = population + 0.5 * K * (population[perm1] - population[perm2])
    
    # Enforce bounds [0, 1]
    new_population = np.clip(new_population, 0.0, 1.0)
    
    # Apply polynomial mutation to ensure fine convergence and diversity
    new_population = polynomial_mutation(new_population)
    
    return new_population

# ------------------------------------------------------------------------------
# 5. Main MOCS Algorithm with NSGA-II Environment Selection
# ------------------------------------------------------------------------------
def run_mocs_dtlz1(pop_size=200, max_gen=1000, num_objectives=3, num_vars=7, pa=0.25):
    """
    Main loop for MOCS solving DTLZ1.
    """
    print(f"Starting MOCS on DTLZ1 (Objectives: {num_objectives}, Variables: {num_vars})")
    print(f"Population: {pop_size}, Generations: {max_gen}")
    
    # Initialize population randomly
    population = np.random.rand(pop_size, num_vars)
    
    # Evaluate initial population
    objectives = np.array([dtlz1(ind, num_objectives) for ind in population])
    
    for gen in range(max_gen):
        # Calculate dynamic decaying alpha (from 0.1 down to 0.01)
        current_alpha = 0.1 * (1.0 - gen / max_gen) + 0.01

        # 1. Non-dominated sort to find the best solutions (first front)
        fronts = fast_non_dominated_sort(objectives)
        best_indices = fronts[0]
        best_solutions = population[best_indices]
        
        # 2. Generate new solutions via Lévy flights (Global Walk) with decaying alpha
        new_pop_levy = get_cuckoos(population, best_solutions, alpha=current_alpha)
        new_obj_levy = np.array([dtlz1(ind, num_objectives) for ind in new_pop_levy])
        
        # 3. Combine and Select (like NSGA-II)
        combined_pop = np.vstack((population, new_pop_levy))
        combined_obj = np.vstack((objectives, new_obj_levy))
        
        # Non-dominated sort on combined
        fronts_comb = fast_non_dominated_sort(combined_obj)
        
        next_pop_indices = []
        for front in fronts_comb:
            if len(next_pop_indices) + len(front) <= pop_size:
                next_pop_indices.extend(front)
            else:
                # Crowding distance sorting for the last front to include
                distances = calculate_crowding_distance(combined_obj, front)
                sorted_front = [x for _, x in sorted(zip(distances, front), reverse=True)]
                num_needed = pop_size - len(next_pop_indices)
                next_pop_indices.extend(sorted_front[:num_needed])
                break
                
        population = combined_pop[next_pop_indices]
        objectives = combined_obj[next_pop_indices]
        
        # 4. Discover and replace nests (Local Walk)
        new_pop_local = empty_nests(population, pa)
        new_obj_local = np.array([dtlz1(ind, num_objectives) for ind in new_pop_local])
        
        # 5. Combine and Select again (to maintain strict elitism)
        combined_pop2 = np.vstack((population, new_pop_local))
        combined_obj2 = np.vstack((objectives, new_obj_local))
        
        fronts_comb2 = fast_non_dominated_sort(combined_obj2)
        
        next_pop_indices2 = []
        for front in fronts_comb2:
            if len(next_pop_indices2) + len(front) <= pop_size:
                next_pop_indices2.extend(front)
            else:
                distances = calculate_crowding_distance(combined_obj2, front)
                sorted_front = [x for _, x in sorted(zip(distances, front), reverse=True)]
                num_needed = pop_size - len(next_pop_indices2)
                next_pop_indices2.extend(sorted_front[:num_needed])
                break
                
        population = combined_pop2[next_pop_indices2]
        objectives = combined_obj2[next_pop_indices2]
        
        if (gen + 1) % 20 == 0:
            print(f"Generation {gen + 1}/{max_gen} completed.")

    print("Optimization finished.")
    return population, objectives

# ------------------------------------------------------------------------------
# 6. Execution and Visualization
# ------------------------------------------------------------------------------
def generate_reference_points(num_objectives, p):
    """
    Generate uniform reference points on a simplex (Das and Dennis approach).
    For DTLZ1, the points sum to 0.5.
    """
    import itertools
    points = []
    for c in itertools.combinations(range(p + num_objectives - 1), num_objectives - 1):
        c = (-1,) + c + (p + num_objectives - 1,)
        point = tuple((c[i+1] - c[i] - 1) / p for i in range(num_objectives))
        points.append(point)
    return np.array(points) * 0.5

if __name__ == "__main__":
    np.random.seed(42)
    
    # Standard DTLZ1 parameters
    num_obj = 3
    num_variables = 7 # k = 5, n = 3 + 5 - 1 = 7
    
    # Run MOCS with increased generations and population
    final_pop, final_obj = run_mocs_dtlz1(pop_size=200, max_gen=1000, num_objectives=num_obj, num_vars=num_variables)
    
    # Get only the non-dominated solutions from the final population
    fronts = fast_non_dominated_sort(final_obj)
    pareto_front_obj = final_obj[fronts[0]]
    
    # Generate reference points to plot lightly in the background for comparison
    ref_points = generate_reference_points(num_objectives=num_obj, p=13)
    
    # Plotting
    fig = plt.figure(figsize=(8, 8))
    # Match the background color of the figure and axes to the reference
    fig.patch.set_facecolor('#D3D3D3') 
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#D3D3D3')
    
    # Plot the true reference points in light gray/white for comparison
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], 
               c='white', marker='o', s=10, alpha=0.5, label='True Front')

    # Plot the MOCS Pareto front in blue like the reference image
    ax.scatter(pareto_front_obj[:, 0], pareto_front_obj[:, 1], pareto_front_obj[:, 2], 
               c='#005596', marker='o', s=30, alpha=1.0, label='MOCS Front')
    
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    ax.set_zlabel('$f_3$')
    
    # Set limits to [0.0, 0.5] if the algorithm converged well, 
    # but allow it to autoscale if it didn't so we can still see the points.
    max_val = np.max(pareto_front_obj)
    upper_limit = max(0.5, max_val * 1.1)
    ax.set_xlim(0.0, upper_limit)
    ax.set_ylim(0.0, upper_limit)
    ax.set_zlim(0.0, upper_limit)
    
    # Adjust view angle to match the reference image
    ax.view_init(elev=35, azim=45)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('mocs_dtlz1_result.png', facecolor=fig.get_facecolor(), edgecolor='none')
    print("Saved MOCS visualization to 'mocs_dtlz1_result.png'.")
