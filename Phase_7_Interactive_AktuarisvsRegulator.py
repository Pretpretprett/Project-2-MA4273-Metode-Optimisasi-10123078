import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from desdeo_problem import DiscreteDataProblem
from desdeo_mcdm.interactive import NIMBUS

def main():
    print("FASE 7: Simulasi Interaktif (Aktuaris vs Regulator) dengan DESDEO NIMBUS")
    
    # 1. Load Data
    df = pd.read_csv('pareto_front_cs_nsga2.csv')
    objective_names = ['f1_RMSE', 'f2_DIR', 'f3_Lip95']
    variable_names = ['w_MO', 'w_MDF', 'w_MSCM']

    ideal = df[objective_names].min().values
    nadir = df[objective_names].max().values

    # Create Discrete problem
    problem = DiscreteDataProblem(df, variable_names, objective_names, ideal, nadir)
    
    # Skenario 1: Preferensi Aktuaris
    
    print("\n--- Skenario 1: Preferensi AKTUARIS ---")
  
    
    method_act = NIMBUS(problem)
    reqs_act = method_act.start()
    req_act = reqs_act[0]
    
    # Actuary wants to improve f1 (<), relax f2 (>=), relax f3 (>=)
    req_act.response = {
        'classifications': ['<', '>=', '>='],
        'levels': [ideal[0], nadir[1], nadir[2]],
        'number_of_solutions': 1
    }
    new_reqs_act = method_act.iterate(req_act)
    act_solution = new_reqs_act[0].content['solutions'][0]
    act_objective = new_reqs_act[0].content['objectives'][0]
    
    print(f"Hasil Pilihan NIMBUS (Aktuaris):")
    print(f"Decision Variables (Bobot): w_MO={act_solution[0]:.4f}, w_MDF={act_solution[1]:.4f}, w_MSCM={act_solution[2]:.4f}")
    print(f"Objective Functions:")
    print(f"  f1 (RMSE):   {act_objective[0]:.4f}")
    print(f"  f2 (|1-DIR|): {act_objective[1]:.4f}")
    print(f"  f3 (Lip95):   {act_objective[2]:.4f}")

   
    # Skenario 2: Preferensi Regulator
   
    print("\n--- Skenario 2: Preferensi REGULATOR ---")
  
    
    method_reg = NIMBUS(problem)
    reqs_reg = method_reg.start()
    req_reg = reqs_reg[0]
    
    # Regulator wants to relax f1 (>=), improve f2 (<), improve f3 (<)
    req_reg.response = {
        'classifications': ['>=', '<', '<'],
        'levels': [nadir[0], ideal[1], ideal[2]],
        'number_of_solutions': 1
    }
    new_reqs_reg = method_reg.iterate(req_reg)
    reg_solution = new_reqs_reg[0].content['solutions'][0]
    reg_objective = new_reqs_reg[0].content['objectives'][0]
    
    print(f"Hasil Pilihan NIMBUS (Regulator):")
    print(f"Decision Variables (Bobot): w_MO={reg_solution[0]:.4f}, w_MDF={reg_solution[1]:.4f}, w_MSCM={reg_solution[2]:.4f}")
    print(f"Objective Functions:")
    print(f"  f1 (RMSE):   {reg_objective[0]:.4f}")
    print(f"  f2 (|1-DIR|): {reg_objective[1]:.4f}")
    print(f"  f3 (Lip95):   {reg_objective[2]:.4f}")

  
    # 3. Visualisasi Simplex Plot (Aktuaris vs Regulator)
 
    print("\nGenerating Barycentric Simplex Plot...")
    
    # Barycentric coordinates transformation
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

    def to_barycentric(w):
        return w[:, 0:1] * corners[0] + w[:, 1:2] * corners[1] + w[:, 2:3] * corners[2]

    W = df[['w_MO', 'w_MDF', 'w_MSCM']].values
    xy = to_barycentric(W)

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot all 1000 pareto solutions as background
    ax.scatter(xy[:, 0], xy[:, 1], c='lightgray', alpha=0.5, s=20, label='Semua Solusi Pareto')

    # Plot Actuary choice
    xy_act = to_barycentric(np.array([act_solution]))
    ax.scatter(xy_act[:, 0], xy_act[:, 1], color='blue', marker='o', s=200, edgecolors='black', zorder=5, label='Pilihan Aktuaris')

    # Plot Regulator choice
    xy_reg = to_barycentric(np.array([reg_solution]))
    ax.scatter(xy_reg[:, 0], xy_reg[:, 1], color='red', marker='s', s=200, edgecolors='black', zorder=5, label='Pilihan Regulator')

    # Draw triangle borders
    ax.plot(corners[[0, 1, 2, 0], 0], corners[[0, 1, 2, 0], 1], 'k-', lw=2)

    ax.text(corners[0, 0] - 0.05, corners[0, 1] - 0.05, 'MO (1, 0, 0)', fontsize=12, ha='center')
    ax.text(corners[1, 0] + 0.05, corners[1, 1] - 0.05, 'MDF (0, 1, 0)', fontsize=12, ha='center')
    ax.text(corners[2, 0], corners[2, 1] + 0.05, 'MSCM (0, 0, 1)', fontsize=12, ha='center')

    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_title('Preferensi Bobot: Aktuaris vs Regulator\n(Barycentric Simplex Plot)', pad=30)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('simplex_interactive.png')
    plt.close()
    print("Plot disimpan sebagai simplex_interactive.png")

if __name__ == '__main__':
    main()
