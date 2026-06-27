import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from desdeo_problem import DiscreteDataProblem
from desdeo_mcdm.interactive import NIMBUS
import warnings

# Suppress annoying numpy/pandas warnings
warnings.filterwarnings("ignore")

def main():
    print("\nLoading Data Pareto...")
    df = pd.read_csv('pareto_front_cs_nsga2.csv')
    
    print(f"Total solusi Pareto Feasible: {len(df)}")
    
    objective_names = ['f1_RMSE', 'f2_DIR', 'f3_Lip95']
    variable_names = ['w_MO', 'w_MDF', 'w_MSCM']

    ideal = df[objective_names].min().values
    nadir = df[objective_names].max().values

    problem = DiscreteDataProblem(df, variable_names, objective_names, ideal, nadir)
    
    saved_solutions = []
  
    print("      INTERACTIVE MCDM - DESDEO NIMBUS (Real-Time)     ")
    print("Panduan Klasifikasi (NIMBUS):")
    print(" '<'  : Objektif ini harus MEMBAIK (turun)")
    print(" '<=' : Boleh membaik atau tetap")
    print(" '='  : Pertahankan nilai saat ini")
    print(" '>=' : Boleh memburuk (naik)")
    print(" '0'  : Bebas berubah tanpa batas")
    
    iteration = 1
    
    while True:
        # Re-initialize method per iteration for a clean state in discrete picking
        method = NIMBUS(problem)
        reqs = method.start()
        req = reqs[0]
        
        print(f"\n--- Iterasi Pencarian #{iteration} ---")
        
        while True:
            c1 = input("Klasifikasi untuk f1 (RMSE Akurasi)           [<, <=, =, >=, 0] : ").strip()
            if c1 in ['<', '<=', '=', '>=', '0']: break
            print("Input tidak valid. Harap masukkan <, <=, =, >=, atau 0")
            
        while True:
            c2 = input("Klasifikasi untuk f2 (Group Fairness)         [<, <=, =, >=, 0] : ").strip()
            if c2 in ['<', '<=', '=', '>=', '0']: break
            print("Input tidak valid. Harap masukkan <, <=, =, >=, atau 0")
            
        while True:
            c3 = input("Klasifikasi untuk f3 (Individual Fairness)    [<, <=, =, >=, 0] : ").strip()
            if c3 in ['<', '<=', '=', '>=', '0']: break
            print("Input tidak valid. Harap masukkan <, <=, =, >=, atau 0")
        
        classifications = [c1, c2, c3]
        
        # Determine levels automatically
        levels = []
        for i, c in enumerate(classifications):
            if c in ['<', '<=']:
                levels.append(ideal[i])
            elif c == '>=':
                levels.append(nadir[i])
            else:
                levels.append(ideal[i]) # fallback for '=' or '0'
        
        req.response = {
            'classifications': classifications,
            'levels': levels,
            'number_of_solutions': 1
        }
        
        try:
            new_reqs = method.iterate(req)
            sol_content = new_reqs[0].content
            
            chosen_w = sol_content['solutions'][0]
            chosen_f = sol_content['objectives'][0]
            
            print("\n✓ Hasil Rekomendasi Algoritma:")
            print(f"  RMSE             : {chosen_f[0]:.4f}")
            print(f"  |1-DIR| penalty  : {chosen_f[1]:.4f}")
            print(f"  Lipschitz        : {chosen_f[2]:.2f}")
            print(f"  Bobot (MO, MDF, MSCM) : [{chosen_w[0]:.8f} {chosen_w[1]:.8f} {chosen_w[2]:.8f}]")
            
            while True:
                save_ans = input("\nApakah Anda puas dan ingin menyimpan hasil ini ke plot? (y/n): ").strip().lower()
                if save_ans in ['y', 'n']: break
            
            if save_ans == 'y':
                name = input("Beri nama untuk persona/keputusan ini (misal: 'Manager'): ").strip()
                saved_solutions.append({
                    'name': name,
                    'w': chosen_w,
                    'f': chosen_f
                })
                print(f"-> Keputusan '{name}' berhasil disimpan!")
                
            while True:
                cont_ans = input("Ingin mencoba kombinasi klasifikasi lain? (y/n): ").strip().lower()
                if cont_ans in ['y', 'n']: break
                
            if cont_ans == 'n':
                break
                
            iteration += 1
            
        except Exception as e:
            print(f"\n[ERROR] Terjadi kesalahan perhitungan: {e}")
            print("Coba lagi dengan kombinasi berbeda.")
            
    print("\nSelesai! Menyimpan dan mem-plot hasil pilihan Anda...")
    
    if len(saved_solutions) > 0:
        # Generate Plot
        corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        def to_barycentric(w):
            return w[:, 0:1] * corners[0] + w[:, 1:2] * corners[1] + w[:, 2:3] * corners[2]
            
        W = df[['w_MO', 'w_MDF', 'w_MSCM']].values
        xy = to_barycentric(W)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(xy[:, 0], xy[:, 1], c='lightgray', alpha=0.5, s=20, label='Pareto Solutions')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        for i, sol in enumerate(saved_solutions):
            w_array = np.array([sol['w']])
            xy_point = to_barycentric(w_array)
            ax.scatter(xy_point[:, 0], xy_point[:, 1], color=colors[i%len(colors)], 
                       marker=markers[i%len(markers)], s=200, edgecolors='black', zorder=5, label=f"Pilihan: {sol['name']}")
                       
        vertices = np.array([[0,0], [1,0], [0.5, np.sqrt(3)/2], [0,0]])
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
        ax.text(vertices[0, 0] - 0.05, vertices[0, 1] - 0.05, 'MO (1, 0, 0)', fontsize=12, ha='center')
        ax.text(vertices[1, 0] + 0.05, vertices[1, 1] - 0.05, 'MDF (0, 1, 0)', fontsize=12, ha='center')
        ax.text(vertices[2, 0], vertices[2, 1] + 0.05, 'MSCM (0, 0, 1)', fontsize=12, ha='center', fontweight='bold')
        
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title('Interactive Real-Time Decision Making\n(Barycentric Simplex Plot)', pad=30)
        
        # Adjust legend to be outside if there are many saved solutions
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('simplex_realtime.png')
        plt.close()
        print("\nPlot disimpan sebagai 'simplex_realtime.png'!")
    else:
        print("\nTidak ada keputusan yang disimpan, plot tidak dibuat.")

if __name__ == "__main__":
    main()
