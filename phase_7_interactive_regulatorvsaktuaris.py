import numpy as np
import pandas as pd
from desdeo_problem.problem import DiscreteDataProblem
from desdeo_mcdm.interactive.NIMBUS import NIMBUS
import sys
import matplotlib.pyplot as plt

def get_barycentric(w):
    # Titik sudut segitiga sama sisi
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, np.sqrt(3)/2])
    return w[0]*A + w[1]*B + w[2]*C

print("FASE 7: Interactive Decision Making dengan DESDEO")

# 1. Load Pareto Front dari Fase 5
try:
    df = pd.read_csv("pareto_front_final.csv")
    # Pastikan data type adalah float
    for col in df.columns:
        df[col] = df[col].astype(float)
except FileNotFoundError:
    print("[ERROR] File 'pareto_front_final.csv' tidak ditemukan.")
    print("Harap eksekusi ulang 'python3 phase_5_6_analysis.py' terlebih dahulu agar data CSV-nya di-generate.")
    sys.exit(1)

# Pastikan data ter-load dengan baik
pareto_f = df[['f1_RMSE', 'f2_DIR_penalty', 'f3_Lipschitz']].values
weights = df[['w_MO', 'w_MDF', 'w_MSCM']].values

ideal = np.min(pareto_f, axis=0)
nadir = np.max(pareto_f, axis=0)

# Inisialisasi DiscreteDataProblem untuk NIMBUS
problem = DiscreteDataProblem(
    df, 
    ['w_MO', 'w_MDF', 'w_MSCM'], 
    ['f1_RMSE', 'f2_DIR_penalty', 'f3_Lipschitz'], 
    ideal=ideal, 
    nadir=nadir
)

print(f"Jumlah solusi di Pareto Front : {len(pareto_f)}")
print(f"Titik Ideal (Terbaik f1,f2,f3)  : {ideal}")
print(f"Titik Nadir (Terburuk f1,f2,f3) : {nadir}\n")


# Sesi 1: Persona DM Regulator (Fokus pada Keadilan)

print("Simulasi DM: REGULATOR")
print("Karakteristik: Mengutamakan Group Fairness (|1-DIR| membaik), rela Akurasi (RMSE) memburuk.")
method_regulator = NIMBUS(problem)
req_reg, _ = method_regulator.start()

# '<' : Harus membaik (turun)
# '<=': Boleh membaik atau tetap
# '=' : Pertahankan
# '>=': Boleh memburuk
# '>f': Boleh memburuk sampai batas tertentu
response_reg = {
    "classifications": [">=", "<", "<="], # f1: RMSE (>=), f2: DIR (<), f3: Lip (<=)
    "number_of_solutions": 1,
    "levels": nadir.tolist() # Valid levels are required between ideal and nadir
}
req_reg.response = response_reg
out_reg, _ = method_regulator.iterate(req_reg)

w_regulator = None
if out_reg.content['objectives'] is not None and len(out_reg.content['objectives']) > 0:
    obj_reg = out_reg.content['objectives'][0]
    # Cari indeks bobot asli (mencari solusi terdekat di dataset)
    idx = np.argmin(np.sum((pareto_f - obj_reg)**2, axis=1))
    w_regulator = weights[idx]
    print(f"  Keputusan Final DM Regulator:")
    print(f"  Bobot (MO, MDF, MSCM) : {w_regulator}")
    print(f"  RMSE                  : {obj_reg[0]:.2f}")
    print(f"  |1-DIR| penalty       : {obj_reg[1]:.4f}")
    print(f"  Lipschitz             : {obj_reg[2]:.2f}\n")
else:
    print("X Tidak ditemukan solusi yang cocok untuk Regulator.\n")


# Sesi 2: Persona DM Aktuaris (Fokus pada Akurasi/Bisnis)

print("Simulasi DM: AKTUARIS")
print("Karakteristik: Mengutamakan Akurasi Bisnis (RMSE membaik), rela keadilan (Fairness) sedikit melonggar.")
method_actuary = NIMBUS(problem)
req_act, _ = method_actuary.start()

response_act = {
    "classifications": ["<", ">=", ">="], # f1: RMSE (<), f2: DIR (>=), f3: Lip (>=)
    "number_of_solutions": 1,
    "levels": nadir.tolist()
}
req_act.response = response_act
out_act, _ = method_actuary.iterate(req_act)

w_actuary = None
if out_act.content['objectives'] is not None and len(out_act.content['objectives']) > 0:
    obj_act = out_act.content['objectives'][0]
    idx = np.argmin(np.sum((pareto_f - obj_act)**2, axis=1))
    w_actuary = weights[idx]
    print(f"  Keputusan Final DM Aktuaris:")
    print(f"  Bobot (MO, MDF, MSCM) : {w_actuary}")
    print(f"  RMSE                  : {obj_act[0]:.2f}")
    print(f"  |1-DIR| penalty       : {obj_act[1]:.4f}")
    print(f"  Lipschitz             : {obj_act[2]:.2f}\n")
else:
    print("X Tidak ditemukan solusi yang cocok untuk Aktuaris.\n")



# Sesi 3: Visualisasi Akhir di Simplex Plot

if w_regulator is not None and w_actuary is not None:
    print("Menyimpan perbandingan visual ke 'decision_maker_simplex.png'...")
    
    plt.figure(figsize=(8, 7))
    # Gambar outline segitiga simplex
    A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3)/2])
    triangle = np.vstack([A, B, C, A])
    plt.plot(triangle[:, 0], triangle[:, 1], 'k-', alpha=0.3)
    
    # Plot semua solusi di pareto front sebagai background
    pts_2d = np.array([get_barycentric(w) for w in weights])
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c='lightgray', s=20, alpha=0.5, label='Pareto Front (CS-NSGA-II)')
    
    # Plot DM decisions
    pt_reg = get_barycentric(w_regulator)
    plt.scatter(pt_reg[0], pt_reg[1], c='blue', marker='*', s=300, edgecolor='black', label='DM Regulator (Fokus Fair)')
    
    pt_act = get_barycentric(w_actuary)
    plt.scatter(pt_act[0], pt_act[1], c='red', marker='P', s=200, edgecolor='black', label='DM Aktuaris (Fokus Akurat)')
    
    # Anotasi sudut segitiga (Base Models)
    plt.text(A[0], A[1]-0.05, 'MO (1,0,0)', ha='center', fontweight='bold')
    plt.text(B[0], B[1]-0.05, 'MDF (0,1,0)', ha='center', fontweight='bold')
    plt.text(C[0], C[1]+0.05, 'MSCM (0,0,1)', ha='center', fontweight='bold')
    
   
    plt.axis('off')
    plt.legend(loc='lower left', bbox_to_anchor=(0.8, 0.8))
    plt.tight_layout()
    plt.savefig("decision_maker_simplex.png", dpi=300)
    print("Simplex plot berhasil disimpan!")
