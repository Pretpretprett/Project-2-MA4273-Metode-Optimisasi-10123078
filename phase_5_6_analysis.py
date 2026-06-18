import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import modul Fase 3 dan 4
from phase_3_objectives import evaluate_objectives, precompute_gower_nn, compute_cv
import importlib.util
spec = importlib.util.spec_from_file_location("phase_4_ns_cs", "phase_4_ns+cs.py")
phase_4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase_4)
run_mocs_insurance = phase_4.run_mocs_insurance

print("FASE 5: Optimisasi CS-NSGA-II ")
print("1. Membaca data prediksi Fase 1 & 2")
df = pd.read_csv("phase12_predictions.csv")

Y_actual = df['PremTot_actual'].values
D_gender = df['D_gender_binary'].values

Y_MO = df['pred_MO'].values
Y_MDF = df['pred_MDF'].values
Y_MSCM = df['pred_MSCM'].values
Y_MU = df['pred_MU'].values
Y_preds = (Y_MO, Y_MDF, Y_MSCM)

rmse_mu = np.sqrt(np.mean((Y_actual - Y_MU) ** 2))
X_test = df.drop(columns=['PremTot_actual', 'D_gender_binary', 'pred_MO', 'pred_MDF', 'pred_MSCM', 'pred_MU'])

print("2. Precompute Gower Distance")

nn_indices, nn_distances = precompute_gower_nn(X_test, subset_size=3000)

def optimization_eval_func(w):
   
    F = evaluate_objectives(w, Y_actual, Y_preds, D_gender, nn_indices, nn_distances)
    cv = compute_cv(w, Y_actual, Y_preds, D_gender, rmse_mu)
    return F, cv

# Eksekusi algoritma utama 
history, final_pop, final_obj, final_cv = run_mocs_insurance(
    eval_func=optimization_eval_func,
    pop_size=200,   
    max_gen=1000,   
    num_objectives=3,
    num_vars=3,
    pa=0.25
)

# Analisis Hasil Akhir
last_pareto_w, last_pareto_F, last_pareto_CV = history[-1]

print("HASIL AKHIR (Pareto Front Terakhir)")
print(f"Total titik di Pareto Front: {len(last_pareto_w)}")

feasible_indices = np.where(last_pareto_CV == 0)[0]
print(f"Total solusi feasible (Memenuhi Constraint g1 & g2): {len(feasible_indices)}")

if len(feasible_indices) > 0:
    print("\n[INFO] Solusi Feasible Terbaik (Sampel 5 teratas):")
    for idx in feasible_indices[:5]:
        w = last_pareto_w[idx]
        f = last_pareto_F[idx]
        print(f"w=(MO:{w[0]:.3f}, MDF:{w[1]:.3f}, MSCM:{w[2]:.3f}) | RMSE: {f[0]:.2f} | DIR: {abs(1-f[1]):.4f} | Lipschitz: {f[2]:.2f}")
else:
    print("\nTidak ada solusi feasible ditemukan. Semua melanggar batas CV > 0.")


# Visualisasi Pareto Front (3D Plot)

print("\n3. Menyimpan Visualisasi Pareto Front")
if len(feasible_indices) > 0:
    # Hanya plot yang feasible
    plot_F = last_pareto_F[feasible_indices]
    title_suffix = "(Feasible Solutions)"
else:
    # Plot semua solusi jika tidak ada yang feasible
    plot_F = last_pareto_F
    title_suffix = "(All Solutions)"

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Warna scatter berdasarkan nilai RMSE (f1)
scatter = ax.scatter(plot_F[:, 0], plot_F[:, 1], plot_F[:, 2], 
                     c=plot_F[:, 0], cmap='viridis', marker='o', s=50, alpha=0.8, label='CS-NSGA-II Front')

ax.set_xlabel('f1 (RMSE)')
ax.set_ylabel('f2 (|1 - DIR|)')
ax.set_zlabel('f3 (Lipschitz Q95)')
ax.set_title(f'Pareto Front 3D - Fairness Insurance {title_suffix}')

fig.colorbar(scatter, ax=ax, label='RMSE (Lower is Better)')
ax.view_init(elev=30, azim=45)
plt.legend()
plt.tight_layout()

plot_filename = "pareto_front_insurance.png"
plt.savefig(plot_filename, dpi=300)
print(f"[INFO] Grafik Pareto Front berhasil disimpan ke file: {plot_filename}")


# Visualisasi 2D: Trade-off antar Objektif

fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Akurasi (RMSE) vs Individual Fairness (Lipschitz)
axes[0].scatter(plot_F[:, 0], plot_F[:, 2], c='#005596', alpha=0.7)
axes[0].set_xlabel('f1 (RMSE) - Lower is Better')
axes[0].set_ylabel('f3 (Lipschitz Q95) - Lower is Better')
axes[0].set_title('Akurasi vs Individual Fairness')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Individual Fairness (Lipschitz) vs Group Fairness (|1-DIR|)
axes[1].scatter(plot_F[:, 2], plot_F[:, 1], c='#005596', alpha=0.7)
axes[1].set_xlabel('f3 (Lipschitz Q95) - Lower is Better')
axes[1].set_ylabel('f2 (|1 - DIR|) - Lower is Better')
axes[1].set_title('Individual vs Group Fairness')
axes[1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Group Fairness (|1-DIR|) vs Akurasi (RMSE)
axes[2].scatter(plot_F[:, 1], plot_F[:, 0], c='#005596', alpha=0.7)
axes[2].set_xlabel('f2 (|1 - DIR|) - Lower is Better')
axes[2].set_ylabel('f1 (RMSE) - Lower is Better')
axes[2].set_title('Group Fairness vs Akurasi')
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plot2d_filename = "pareto_front_2d_tradeoffs.png"
plt.savefig(plot2d_filename, dpi=300)
print(f"[INFO] Grafik 2D Trade-offs berhasil disimpan ke file: {plot2d_filename}")

# Simpan Pareto Front ke CSV untuk digunakan di Fase 7
pareto_df = pd.DataFrame({
    'w_MO': plot_F[:, 0] * 0, # Placeholder, will fill correctly below
    'w_MDF': plot_F[:, 1] * 0,
    'w_MSCM': plot_F[:, 2] * 0,
    'f1_RMSE': plot_F[:, 0],
    'f2_DIR_penalty': plot_F[:, 1],
    'f3_Lipschitz': plot_F[:, 2]
})

if len(feasible_indices) > 0:
    pareto_df['w_MO'] = last_pareto_w[feasible_indices, 0]
    pareto_df['w_MDF'] = last_pareto_w[feasible_indices, 1]
    pareto_df['w_MSCM'] = last_pareto_w[feasible_indices, 2]
else:
    pareto_df['w_MO'] = last_pareto_w[:, 0]
    pareto_df['w_MDF'] = last_pareto_w[:, 1]
    pareto_df['w_MSCM'] = last_pareto_w[:, 2]

pareto_df.to_csv("pareto_front_final.csv", index=False)
print("[INFO] Data Pareto Front berhasil disimpan ke 'pareto_front_final.csv'")