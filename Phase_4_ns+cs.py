import numpy as np
import pandas as pd
import gower
import time
import math
from Phase_3_objectives import compute_rmse, compute_dir, compute_lipschitz, evaluate_objectives


# 1. Simplex Projection & Initialization

def project_to_simplex(v):
 
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def random_simplex_population(pop_size, num_vars):
  
    return np.random.dirichlet(np.ones(num_vars), size=pop_size)


# 2. Non-Dominated Sorting and Crowding Distance

def fast_non_dominated_sort(objectives):
    pop_size = objectives.shape[0]
    S = [[] for _ in range(pop_size)]
    front = [[]]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size, dtype=int)

    for p in range(pop_size):
        S[p] = []
        n[p] = 0
        for q in range(pop_size):
            # Check if p dominates q (minimization)
            if (objectives[p] <= objectives[q]).all() and (objectives[p] < objectives[q]).any():
                S[p].append(q)
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

def calculate_crowding_distance(objectives, front):
    num_solutions = len(front)
    distances = np.zeros(num_solutions)
    
    if num_solutions == 0:
        return distances
    if num_solutions <= 2:
        return np.full(num_solutions, np.inf)

    num_objectives = objectives.shape[1]
    
    for m in range(num_objectives):
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


# 3. Cuckoo Search Operators (Lévy Flight)

def get_levy_flight_step(beta, size):
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
               (math.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))))**(1 / beta)
    sigma_v = 1.0

    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)
    
    step = u / (np.abs(v)**(1 / beta))
    return step

def get_cuckoos_simplex(population, best_solutions, beta=1.5, alpha=0.01):
    pop_size, num_vars = population.shape
    new_population = np.zeros_like(population)
    
    for i in range(pop_size):
        best = best_solutions[np.random.randint(0, len(best_solutions))]
        step = get_levy_flight_step(beta, num_vars)
        
        new_w = population[i] + alpha * step * (population[i] - best)
        new_population[i] = project_to_simplex(new_w)
        
    return new_population

def empty_nests_simplex(population, pa=0.25):
    pop_size, num_vars = population.shape
    new_population = np.copy(population)
    
    K = np.random.rand(pop_size, num_vars) < pa
    perm1 = np.random.permutation(pop_size)
    perm2 = np.random.permutation(pop_size)
    
    new_w_matrix = population + 0.5 * K * (population[perm1] - population[perm2])
    
    for i in range(pop_size):
        new_population[i] = project_to_simplex(new_w_matrix[i])
        
    return new_population


# 4. Main CS-NSGA-II Algorithm

def evaluate_population(population, y_actual, y_preds, d_gender, nn_indices, d_g):
    objs = []
    for w in population:
        objs.append(evaluate_objectives(w, y_actual, y_preds, d_gender, nn_indices, d_g))
    return np.array(objs)

def run_cs_nsga2(y_actual, y_preds, d_gender, nn_indices, d_g, 
                 pop_size=200, max_gen=50, num_vars=3, pa=0.25, run_id=1):
    print(f"\n--- Run {run_id} ---")
    population = random_simplex_population(pop_size, num_vars)
    objectives = evaluate_population(population, y_actual, y_preds, d_gender, nn_indices, d_g)
    
    history = []
    
    for gen in range(max_gen):
        
        current_alpha = 0.1 * (1.0 - gen / max_gen) + 0.01
        
        # 1. Non-dominated sort
        fronts = fast_non_dominated_sort(objectives)
        best_solutions = population[fronts[0]]
        
        # 2. Lévy Flights
        new_pop_levy = get_cuckoos_simplex(population, best_solutions, alpha=current_alpha)
        new_obj_levy = evaluate_population(new_pop_levy, y_actual, y_preds, d_gender, nn_indices, d_g)
        
        # 3. Combine and Select
        combined_pop = np.vstack((population, new_pop_levy))
        combined_obj = np.vstack((objectives, new_obj_levy))
        
        fronts_comb = fast_non_dominated_sort(combined_obj)
        next_pop_indices = []
        for front in fronts_comb:
            if len(next_pop_indices) + len(front) <= pop_size:
                next_pop_indices.extend(front)
            else:
                distances = calculate_crowding_distance(combined_obj, front)
                sorted_front = [x for _, x in sorted(zip(distances, front), reverse=True)]
                num_needed = pop_size - len(next_pop_indices)
                next_pop_indices.extend(sorted_front[:num_needed])
                break
                
        population = combined_pop[next_pop_indices]
        objectives = combined_obj[next_pop_indices]
        
        # 4. Empty Nests (Abandon worst)
        new_pop_local = empty_nests_simplex(population, pa)
        new_obj_local = evaluate_population(new_pop_local, y_actual, y_preds, d_gender, nn_indices, d_g)
        
        # 5. Combine and Select again
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
        
        front0_size = len(fast_non_dominated_sort(objectives)[0])
        
        # Track convergence (min of objectives in front 0) to show monotonic decrease
        front0_objs = objectives[fast_non_dominated_sort(objectives)[0]]
        history.append({
            'gen': gen + 1,
            'f1_min': np.min(front0_objs[:, 0]),
            'f2_min': np.min(front0_objs[:, 1]),
            'f3_min': np.min(front0_objs[:, 2])
        })
        
        if (gen + 1) % 10 == 0:
            print(f"Run {run_id} | Generation {gen + 1}/{max_gen} | Front 0 size: {front0_size}")

    # Return only the non-dominated solutions of the final population
    final_fronts = fast_non_dominated_sort(objectives)
    pareto_indices = final_fronts[0]
    return population[pareto_indices], objectives[pareto_indices], history


# 5. Multi-Run Execution 

if __name__ == "__main__":
    np.random.seed(42)
    print("FASE 4: Implementasi CS-NSGA-II (Multi-Run Pooling)")
    
    print("Loading data...")
    df = pd.read_csv("phase12_predictions.csv")
    y_actual = df['PremTot_actual'].values
    d_gender = df['D_gender_binary'].values
    y_preds = df[['pred_MO', 'pred_MDF', 'pred_MSCM']].values
    feature_cols = [c for c in df.columns if c not in ['D_gender_binary', 'PremTot_actual', 'pred_MO', 'pred_MDF', 'pred_MSCM', 'pred_MU']]
    X_test = df[feature_cols].copy()
    
    print("Computing Gower distance matrix...")
    start_time = time.time()
    gower_matrix = gower.gower_matrix(X_test)
    print(f"Gower matrix computed in {time.time() - start_time:.2f} seconds")
    
    # Precompute nearest neighbors
    N = len(y_actual)
    np.fill_diagonal(gower_matrix, np.inf)
    nn_indices = np.argmin(gower_matrix, axis=1)
    d_g = gower_matrix[np.arange(N), nn_indices]
    d_g[d_g == 0] = 1e-9
    del gower_matrix # Free memory
    
    # User's lecturer requested: Run 7 times with 200 population, then pool
    num_runs = 7
    pop_size = 200
    max_gen = 1000
    
    all_pareto_pop = []
    all_pareto_obj = []
    all_history = []
    
    for r in range(1, num_runs + 1):
        pop_pareto, obj_pareto, hist = run_cs_nsga2(
            y_actual, y_preds, d_gender, nn_indices, d_g, 
            pop_size=pop_size, max_gen=max_gen, run_id=r
        )
        all_pareto_pop.append(pop_pareto)
        all_pareto_obj.append(obj_pareto)
        all_history.append(pd.DataFrame(hist).assign(run=r))
        
    # Save convergence history
    history_df = pd.concat(all_history)
    history_df.to_csv("convergence_history.csv", index=False)
    print("Convergence history saved to 'convergence_history.csv'")
    
    # Combine all 7 runs
    pooled_pop = np.vstack(all_pareto_pop)
    pooled_obj = np.vstack(all_pareto_obj)
    
    print(f"\nTotal solutions pooled from {num_runs} runs: {pooled_pop.shape[0]}")
    
    # Final non-dominated sort across the entire pool
    final_fronts = fast_non_dominated_sort(pooled_obj)
    global_pareto_indices = final_fronts[0]
    
    # Apply Crowding Distance to the pooled Front 0 to smooth it and extract exactly 1000 best points
    if len(global_pareto_indices) > 1000:
        distances = calculate_crowding_distance(pooled_obj, global_pareto_indices)
        # Sort indices by descending distance
        sorted_by_cd = [x for _, x in sorted(zip(distances, global_pareto_indices), reverse=True)]
        final_selected_indices = sorted_by_cd[:1000]
    else:
        final_selected_indices = global_pareto_indices
        
    global_pareto_pop = pooled_pop[final_selected_indices]
    global_pareto_obj = pooled_obj[final_selected_indices]
    
    print(f"Final Non-Dominated Pareto Solutions extracted & smoothed: {global_pareto_pop.shape[0]}")
    
    
    res_df = pd.DataFrame({
        'w_MO': global_pareto_pop[:, 0],
        'w_MDF': global_pareto_pop[:, 1],
        'w_MSCM': global_pareto_pop[:, 2],
        'f1_RMSE': global_pareto_obj[:, 0],
        'f2_DIR': global_pareto_obj[:, 1],
        'f3_Lip95': global_pareto_obj[:, 2]
    })
    
    res_df.to_csv("pareto_front_cs_nsga2.csv", index=False)
    print("Pareto front saved to 'pareto_front_cs_nsga2.csv'")
