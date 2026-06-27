import pandas as pd
import numpy as np
import gower
import time

def compute_rmse(y_actual, w, y_preds):
    y_ens = np.dot(y_preds, w)
    return np.sqrt(np.mean((y_actual - y_ens) ** 2))

def compute_dir(d_gender, w, y_preds):
    y_ens = np.dot(y_preds, w)
    mean_f = np.mean(y_ens[d_gender == 1])
    mean_m = np.mean(y_ens[d_gender == 0])
    
    if mean_m == 0:
        dir_val = 0
    else:
        dir_val = mean_f / mean_m
    return abs(1 - dir_val)

def compute_lipschitz(w, y_preds, nn_indices, d_g):
    y_ens = np.dot(y_preds, w)
    
    y_diff = np.abs(y_ens - y_ens[nn_indices])
    lipschitz_vals = y_diff / d_g
    
    return np.percentile(lipschitz_vals, 95)

def evaluate_objectives(w, y_actual, y_preds, d_gender, nn_indices, d_g):
    f1 = compute_rmse(y_actual, w, y_preds)
    f2 = compute_dir(d_gender, w, y_preds)
    f3 = compute_lipschitz(w, y_preds, nn_indices, d_g)
    return np.array([f1, f2, f3])

if __name__ == "__main__":
    print("FASE 3: Definisi Fungsi Objektif")
    
    try:
        df = pd.read_csv("phase12_predictions.csv")
        
        y_actual = df['PremTot_actual'].values
        d_gender = df['D_gender_binary'].values
        y_preds = df[['pred_MO', 'pred_MDF', 'pred_MSCM']].values
        
        # We need the features X to compute gower distance
        feature_cols = [c for c in df.columns if c not in ['D_gender_binary', 'PremTot_actual', 'pred_MO', 'pred_MDF', 'pred_MSCM', 'pred_MU']]
        X_test = df[feature_cols].copy()
        
        
        start_time = time.time()
        
        # To avoid MemoryError, we can compute it on float32 and check if memory is enough
        # The gower package handles it directly.
        gower_matrix = gower.gower_matrix(X_test)
        print(f"Gower matrix shape: {gower_matrix.shape}, computed in {time.time() - start_time:.2f} seconds")
        
        # Precompute nearest neighbors and distances for Lipschitz
        N = len(y_actual)
        np.fill_diagonal(gower_matrix, np.inf)
        nn_indices = np.argmin(gower_matrix, axis=1)
        d_g = gower_matrix[np.arange(N), nn_indices]
        d_g[d_g == 0] = 1e-9
        del gower_matrix # Free memory
        
        # Test objectives with some weights
        weights_to_test = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1/3, 1/3, 1/3)
        ]
        
        for w in weights_to_test:
            w_arr = np.array(w)
            objs = evaluate_objectives(w_arr, y_actual, y_preds, d_gender, nn_indices, d_g)
            print(f"w = {w} -> [RMSE: {objs[0]:.2f}, |1-DIR|: {objs[1]:.4f}, Lip95: {objs[2]:.4f}]")

    except FileNotFoundError:
        print("phase12_predictions.csv not found! Make sure Phase_1_2_modeling.py has finished.")
