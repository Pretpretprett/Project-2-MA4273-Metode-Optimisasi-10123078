import numpy as np
import pandas as pd
import gower

def get_ensemble_predictions(w, Y_preds):
    return w[0] * Y_preds[0] + w[1] * Y_preds[1] + w[2] * Y_preds[2]

def compute_rmse(Y_actual, w, Y_preds):
    """
    Objektif 1: Akurasi (minimisasi RMSE)
    """
    Y_ens = get_ensemble_predictions(w, Y_preds)
    mse = np.mean((Y_actual - Y_ens) ** 2)
    return np.sqrt(mse)

def compute_dir(D_gender, w, Y_preds):
    """
    Objektif 2: Group Fairness (minimisasi |1 - DIR|)
    D_gender: 1 untuk Female, 0 untuk Male
    """
    Y_ens = get_ensemble_predictions(w, Y_preds)
    mean_f = np.mean(Y_ens[D_gender == 1])
    mean_m = np.mean(Y_ens[D_gender == 0])
    dir_val = mean_f / mean_m if mean_m > 0 else 0
    return abs(1.0 - dir_val)

def precompute_gower_nn(X_data, subset_size=None):
    if subset_size is not None and subset_size < len(X_data):
        print(f"Menggunakan subset data berukuran {subset_size} untuk menghitung Gower...")
        X_data = X_data.iloc[:subset_size].copy()
    else:
        print("Menghitung matriks jarak Gower (mungkin butuh waktu beberapa menit)...")
        
    # Matriks jarak Gower (N x N)
  
    gower_dist = gower.gower_matrix(X_data)
    
    # Isi diagonal dengan nilai sangat besar agar tidak terpilih sebagai NN dari dirinya sendiri
    np.fill_diagonal(gower_dist, np.inf)
    
    # Dapatkan indeks dan jarak nearest neighbor
    nn_indices = np.argmin(gower_dist, axis=1)
    nn_distances = gower_dist[np.arange(len(gower_dist)), nn_indices]
    
    return nn_indices, nn_distances

def compute_lipschitz(w, Y_preds, nn_indices, nn_distances):
    """
    Objektif 3: Individual Fairness (Local Lipschitz)
    Menggunakan persentil ke-95.
    """
    Y_ens = get_ensemble_predictions(w, Y_preds)
    
    # Karena nn_indices mungkin dari subset, pastikan ukuran Y_ens disesuaikan dengan nn_indices
    Y_ens_subset = Y_ens[:len(nn_indices)]
    
    # Selisih prediksi antara tiap individu dan nearest neighbor-nya
    pred_diff = np.abs(Y_ens_subset - Y_ens_subset[nn_indices])
    
    # Hindari pembagian dengan 0
    valid_distances = np.where(nn_distances == 0, 1e-6, nn_distances)
    
    lipschitz_constants = pred_diff / valid_distances
    
    # Mengambil Q95
    return np.percentile(lipschitz_constants, 95)

def evaluate_objectives(w, Y_actual, Y_preds, D_gender, nn_indices, nn_distances):
  
    f1 = compute_rmse(Y_actual, w, Y_preds)
    f2 = compute_dir(D_gender, w, Y_preds)
    f3 = compute_lipschitz(w, Y_preds, nn_indices, nn_distances)
    return np.array([f1, f2, f3])


def g1(w, D_test, Y_matrix, eps_DIR=0.05):
    '''
    Constraint 1: |1 - DIR(w)| - eps_DIR <= 0
    '''
    # compute_dir sudah mengembalikan nilai abs(1 - DIR)
    dir_penalty = compute_dir(D_test, w, Y_matrix)
    return dir_penalty - eps_DIR
    # positif = infeasible (melanggar regulasi)
    # negatif = feasible (comply)

def g2(w, Y_actual, Y_matrix, RMSE_MU, tolerance=1.02):
    '''
    Constraint 2: RMSE(w) - (RMSE_MU * tolerance) <= 0
    Memberikan toleransi (misal 2% / 1.02) agar trade-off fairness bisa tercapai.
    '''
    # Menggunakan fungsi compute_rmse yang sudah ada
    rmse_val = compute_rmse(Y_actual, w, Y_matrix)
    return rmse_val - (RMSE_MU * tolerance)
    # positif = infeasible (lebih buruk dari batas toleransi)
    # negatif = feasible (comply)

def compute_cv(w, Y_actual, Y_preds, D_gender, rmse_mu, eps_DIR=0.05):
    '''
    Total Constraint Violation (CV)
    '''
    val_g1 = g1(w, D_gender, Y_preds, eps_DIR)
    val_g2 = g2(w, Y_actual, Y_preds, rmse_mu)
    return max(0, val_g1) + max(0, val_g2)

if __name__ == "__main__":
    print("Testing Fase 3: Fungsi Objektif")
    
    print("Membaca data hasil Fase 1 & 2 (phase12_predictions.csv)")
    df = pd.read_csv("phase12_predictions.csv")
    
    Y_actual = df['PremTot_actual'].values
    D_gender = df['D_gender_binary'].values
    
    Y_MO = df['pred_MO'].values
    Y_MDF = df['pred_MDF'].values
    Y_MSCM = df['pred_MSCM'].values
    Y_MU = df['pred_MU'].values
    Y_preds = (Y_MO, Y_MDF, Y_MSCM)
    
    # Hitung RMSE_MU
    mse_mu = np.mean((Y_actual - Y_MU) ** 2)
    rmse_mu = np.sqrt(mse_mu)
    print(f"RMSE_MU dihitung: {rmse_mu:.2f}")
    
    X_test = df.drop(columns=['PremTot_actual', 'D_gender_binary', 'pred_MO', 'pred_MDF', 'pred_MSCM', 'pred_MU'])
    
    # Hitung precompute NN dari Gower
    # Gunakan subset sebanyak 3000 data seperti yang disarankan di roadmap untuk efisiensi
    nn_indices, nn_distances = precompute_gower_nn(X_test, subset_size=3000)
    
    # Uji dengan beberapa kombinasi bobot ekstrim
    weights_to_test = [
        (1, 0, 0),     # MO only
        (0, 1, 0),     # MDF only
        (0, 0, 1),     # MSCM only
        (1/3, 1/3, 1/3) # Equal weights
    ]
    
    print("\n[INFO] Hasil Evaluasi Fungsi Objektif & Constraints:")
    print(f"{'Bobot (w1, w2, w3)':<20} | {'f1 (RMSE)':<10} | {'f2 (|1-DIR|)':<12} | {'f3 (Lipschitz)':<14} | {'CV (Violation)':<15}")
    print("-" * 80)
    
    for w in weights_to_test:
        obj = evaluate_objectives(w, Y_actual, Y_preds, D_gender, nn_indices, nn_distances)
        cv = compute_cv(w, Y_actual, Y_preds, D_gender, rmse_mu)
        print(f"{str(tuple(round(x,3) for x in w)):<20} | {obj[0]:<10.2f} | {obj[1]:<12.4f} | {obj[2]:<14.2f} | {cv:<15.4f}")
    
    print("Fungsi objektif berjalan dengan baik.")
