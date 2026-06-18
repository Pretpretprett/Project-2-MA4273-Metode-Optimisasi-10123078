import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desdeo_problem.problem import DiscreteDataProblem
from desdeo_mcdm.interactive.NIMBUS import NIMBUS

def get_barycentric(w):
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, np.sqrt(3)/2])
    return w[0]*A + w[1]*B + w[2]*C


print(" FASE 7: TRUE INTERACTIVE DECISION MAKING (DESDEO NIMBUS) ")


# 1. Load Pareto Front
try:
    df = pd.read_csv("pareto_front_final.csv")
    for col in df.columns:
        df[col] = df[col].astype(float)
except FileNotFoundError:
    print("[ERROR] File 'pareto_front_final.csv' tidak ditemukan. Jalankan Fase 5 dulu.")
    sys.exit(1)

pareto_f = df[['f1_RMSE', 'f2_DIR_penalty', 'f3_Lipschitz']].values
weights = df[['w_MO', 'w_MDF', 'w_MSCM']].values

ideal = np.min(pareto_f, axis=0)
nadir = np.max(pareto_f, axis=0)

problem = DiscreteDataProblem(
    df, 
    ['w_MO', 'w_MDF', 'w_MSCM'], 
    ['f1_RMSE', 'f2_DIR_penalty', 'f3_Lipschitz'], 
    ideal=ideal, 
    nadir=nadir
)

print(f"Total solusi Pareto yang tersedia: {len(pareto_f)}\n")
print(f"Batas Terbaik (Ideal)  -> RMSE: {ideal[0]:.2f} | DIR Penalty: {ideal[1]:.4f} | Lipschitz: {ideal[2]:.2f}")
print(f"Batas Terburuk (Nadir) -> RMSE: {nadir[0]:.2f} | DIR Penalty: {nadir[1]:.4f} | Lipschitz: {nadir[2]:.2f}\n")
print("Panduan Klasifikasi (NIMBUS):")
print("  '<'  : Objektif ini harus MEMBAIK (turun)")
print("  '<=' : Boleh membaik atau tetap")
print("  '='  : Pertahankan nilai saat ini")
print("  '>=' : Boleh memburuk (naik)\n")

history_w = []
history_labels = []

iterasi = 1
while True:
    print(f"--- Iterasi Pencarian #{iterasi} ---")
    
    # Meminta input langsung dari Decision Maker (DM)
    c1 = input("Klasifikasi untuk f1 (RMSE Akurasi)       [<, <=, =, >=] : ").strip()
    c2 = input("Klasifikasi untuk f2 (Group Fairness)     [<, <=, =, >=] : ").strip()
    c3 = input("Klasifikasi untuk f3 (Individual Fairness)[<, <=, =, >=] : ").strip()
    
    # Validasi input sederhana
    valid_symbols = ['<', '<=', '=', '>=']
    if not all(c in valid_symbols for c in [c1, c2, c3]):
        print("[!] Input tidak valid. Harap hanya masukkan simbol <, <=, =, atau >=\n")
        continue

    # Setup NIMBUS
    method = NIMBUS(problem)
    req, _ = method.start()
    
    req.response = {
        "classifications": [c1, c2, c3],
        "number_of_solutions": 1,
        "levels": nadir.tolist()
    }
    
    # Generate solusi
    out, _ = method.iterate(req)
    
    if out.content['objectives'] is not None and len(out.content['objectives']) > 0:
        obj_result = out.content['objectives'][0]
        idx = np.argmin(np.sum((pareto_f - obj_result)**2, axis=1))
        w_result = weights[idx]
        
        print("\n✓ Hasil Rekomendasi Algoritma:")
        print(f"  RMSE                  : {obj_result[0]:.2f}")
        print(f"  |1-DIR| penalty       : {obj_result[1]:.4f}")
        print(f"  Lipschitz             : {obj_result[2]:.2f}")
        print(f"  Bobot (MO, MDF, MSCM) : {w_result}\n")
        
        simpan = input("Apakah Anda puas dan ingin menyimpan hasil ini ke plot? (y/n): ").strip().lower()
        if simpan == 'y':
            label_name = input("Beri nama untuk persona/keputusan ini (misal: 'Manager'): ").strip()
            history_w.append((w_result, label_name))
            print(f"Keputusan '{label_name}' berhasil disimpan ke memori.\n")
    else:
        print("\nX Algoritma tidak dapat menemukan solusi Pareto yang cocok dengan klasifikasi yang terlalu ketat.\n")

    lagi = input("Ingin mencoba kombinasi klasifikasi lain? (y/n): ").strip().lower()
    if lagi != 'y':
        break
    iterasi += 1


# Menyimpan hasil akhir ke Visualisasi Simplex

if len(history_w) > 0:
    print("\nMenyimpan perbandingan visual ke 'interactive_dm_simplex.png'")
    
    plt.figure(figsize=(8, 7))
    A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3)/2])
    triangle = np.vstack([A, B, C, A])
    plt.plot(triangle[:, 0], triangle[:, 1], 'k-', alpha=0.3)
    
    # Background pareto front
    pts_2d = np.array([get_barycentric(w) for w in weights])
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c='lightgray', s=20, alpha=0.5, label='Pareto Front')
    
    # Plot user choices
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['*', 'P', 'X', 'D', 's']
    for i, (w_res, label) in enumerate(history_w):
        pt = get_barycentric(w_res)
        plt.scatter(pt[0], pt[1], c=colors[i % len(colors)], marker=markers[i % len(markers)], 
                    s=250, edgecolor='black', label=label)
    
    plt.text(A[0], A[1]-0.05, 'MO', ha='center', fontweight='bold')
    plt.text(B[0], B[1]-0.05, 'MDF', ha='center', fontweight='bold')
    plt.text(C[0], C[1]+0.05, 'MSCM', ha='center', fontweight='bold')
    
  
    plt.axis('off')
    plt.legend(loc='lower left', bbox_to_anchor=(0.8, 0.8))
    plt.tight_layout()
    plt.savefig("interactive_dm_simplex.png", dpi=300)
    print("Grafik 'interactive_dm_simplex.png' berhasil dibuat! Silakan buka gambarnya.")
else:
    print("\nTidak ada keputusan yang disimpan. Program selesai.")
