import numpy as np
import math
import copy


# 1. Simplex Projection (Khusus untuk Ensemble Weights Asuransi)

def project_to_simplex(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def random_simplex_point(dim=3):
    pts = np.random.exponential(scale=1.0, size=dim)
    return pts / np.sum(pts)


# 2. Constrained Non-Dominated Sorting (Diadaptasi dari NSGA-II)

def dominates(obj_p, cv_p, obj_q, cv_q):
    if cv_p == 0 and cv_q > 0:
        return True
    elif cv_p > 0 and cv_q == 0:
        return False
    elif cv_p > 0 and cv_q > 0:
        return cv_p < cv_q
    else:
        # Keduanya feasible (cv_p == 0 dan cv_q == 0)
        # Check if p dominates q (p <= q for all, p < q for at least one)
        return (obj_p <= obj_q).all() and (obj_p < obj_q).any()

def fast_non_dominated_sort(objectives, cv):
    pop_size = objectives.shape[0]
    S = [[] for _ in range(pop_size)]
    front = [[]]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size, dtype=int)

    for p in range(pop_size):
        S[p] = []
        n[p] = 0
        for q in range(pop_size):
            if p == q: continue
            
            if dominates(objectives[p], cv[p], objectives[q], cv[q]):
                S[p].append(q)
            elif dominates(objectives[q], cv[q], objectives[p], cv[p]):
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


# 3. Crowding Distance Assignment (dari NSGA-II DTLZ)

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


# 4. Cuckoo Search Operators (dari DTLZ, diubah untuk Simplex)

def get_levy_flight_step(beta, size):
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
               (math.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))))**(1 / beta)
    sigma_v = 1.0

    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, sigma_v, size)
    
    step = u / (np.abs(v)**(1 / beta))
    return step

def polynomial_mutation_simplex(population, pm_prob=None, eta_m=20.0):
    pop_size, num_vars = population.shape
    if pm_prob is None:
        pm_prob = 1.0 / num_vars
        
    mutated = np.copy(population)
    for i in range(pop_size):
        mutated_flag = False
        for j in range(num_vars):
            if np.random.rand() < pm_prob:
                u = np.random.rand()
                if u <= 0.5:
                    delta_q = (2*u)**(1.0/(eta_m+1)) - 1.0
                    mutated[i, j] = mutated[i, j] + delta_q * mutated[i, j]
                else:
                    delta_q = 1.0 - (2.0*(1.0-u))**(1.0/(eta_m+1))
                    mutated[i, j] = mutated[i, j] + delta_q * (1.0 - mutated[i, j])
                mutated_flag = True
        
        if mutated_flag:
            mutated[i] = project_to_simplex(mutated[i])
            
    return mutated

def get_cuckoos_simplex(population, best_solutions, beta=1.5, alpha=0.1):
    pop_size, num_vars = population.shape
    new_population = np.zeros_like(population)
    
    for i in range(pop_size):
        best = best_solutions[np.random.randint(0, len(best_solutions))]
        step = get_levy_flight_step(beta, num_vars)
        
        # Global random walk
        new_population[i] = population[i] + alpha * step * (population[i] - best)
        
    
        new_population[i] = project_to_simplex(new_population[i])
        
    new_population = polynomial_mutation_simplex(new_population)
    return new_population

def empty_nests_simplex(population, pa=0.25):
    pop_size, num_vars = population.shape
    new_population = np.copy(population)
    
    K = np.random.rand(pop_size, num_vars) < pa
    
    perm1 = np.random.permutation(pop_size)
    perm2 = np.random.permutation(pop_size)
    
    # Local random walk 
    new_population = population + 0.5 * K * (population[perm1] - population[perm2])
    
    
    for i in range(pop_size):
        new_population[i] = project_to_simplex(new_population[i])
        
    new_population = polynomial_mutation_simplex(new_population)
    return new_population


# 5. Main MOCS Algorithm with NSGA-II Selection (Insurance Version)

def run_mocs_insurance(eval_func, pop_size=200, max_gen=1000, num_objectives=3, num_vars=3, pa=0.25):
   
    print(f"Starting MOCS on Insurance Problem (Pop: {pop_size}, Gen: {max_gen})")
    
    # Initialize population randomly 
    population = np.array([random_simplex_point(dim=num_vars) for _ in range(pop_size)])
    
    # Evaluate initial population
    objectives = np.zeros((pop_size, num_objectives))
    cv_vals = np.zeros(pop_size)
    
    for i in range(pop_size):
        obj, cv = eval_func(population[i])
        objectives[i] = obj
        cv_vals[i] = cv
        
    history_pareto = []
        
    for gen in range(max_gen):
       
        current_alpha = 0.1 * (1.0 - gen / max_gen) + 0.01

        # 1. Non-dominated sort to find the best solutions
        fronts = fast_non_dominated_sort(objectives, cv_vals)
        best_indices = fronts[0]
        best_solutions = population[best_indices]
        
        # 2. Generate new solutions via Lévy flights
        new_pop_levy = get_cuckoos_simplex(population, best_solutions, alpha=current_alpha)
        new_obj_levy = np.zeros((pop_size, num_objectives))
        new_cv_levy = np.zeros(pop_size)
        
        for i in range(pop_size):
            obj, cv = eval_func(new_pop_levy[i])
            new_obj_levy[i] = obj
            new_cv_levy[i] = cv
        
        # 3. Combine and Select
        combined_pop = np.vstack((population, new_pop_levy))
        combined_obj = np.vstack((objectives, new_obj_levy))
        combined_cv = np.concatenate((cv_vals, new_cv_levy))
        
        fronts_comb = fast_non_dominated_sort(combined_obj, combined_cv)
        
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
        cv_vals = combined_cv[next_pop_indices]
        
        # 4. Discover and replace nests
        new_pop_local = empty_nests_simplex(population, pa)
        new_obj_local = np.zeros((pop_size, num_objectives))
        new_cv_local = np.zeros(pop_size)
        
        for i in range(pop_size):
            obj, cv = eval_func(new_pop_local[i])
            new_obj_local[i] = obj
            new_cv_local[i] = cv
        
        # 5. Combine and Select again
        combined_pop2 = np.vstack((population, new_pop_local))
        combined_obj2 = np.vstack((objectives, new_obj_local))
        combined_cv2 = np.concatenate((cv_vals, new_cv_local))
        
        fronts_comb2 = fast_non_dominated_sort(combined_obj2, combined_cv2)
        
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
        cv_vals = combined_cv2[next_pop_indices2]
        
       
        current_fronts = fast_non_dominated_sort(objectives, cv_vals)
        pf_indices = current_fronts[0]
        history_pareto.append((population[pf_indices], objectives[pf_indices], cv_vals[pf_indices]))
        
        if (gen + 1) % 50 == 0:
            feasible_count = np.sum(cv_vals[pf_indices] == 0)
            print(f"Gen {gen + 1}/{max_gen} | Pareto size: {len(pf_indices)} | Feasible: {feasible_count}")

    print("Optimization finished.")
    return history_pareto, population, objectives, cv_vals

if __name__ == "__main__":
    print("Modul CS-NSGA-II diimpor atau dieksekusi secara stand-alone.")
    print("Silakan gunakan fungsi run_mocs_insurance() dengan menyediakan fungsi evaluasi.")
