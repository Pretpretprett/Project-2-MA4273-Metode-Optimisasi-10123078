import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    print("FASE 5: Optimisasi & Pareto Front")
    df = pd.read_csv('pareto_front_cs_nsga2.csv')

    # Objectives
    F = df[['f1_RMSE', 'f2_DIR', 'f3_Lip95']].values
    weights = df[['w_MO', 'w_MDF', 'w_MSCM']].values

    
    omega = np.array([0.4, 0.35, 0.25])

    # Normalize
    norm_F = F / np.sqrt(np.sum(F**2, axis=0))

    v = norm_F * omega

    A_plus = np.min(v, axis=0)
    A_minus = np.max(v, axis=0)

    D_plus = np.sqrt(np.sum((v - A_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((v - A_minus)**2, axis=1))


    C = D_minus / (D_plus + D_minus)

    # Select best
    best_idx = np.argmax(C)
    w_star = weights[best_idx]
    f_star = F[best_idx]

    print(f"w* (TOPSIS): {w_star}")
    print(f"Objective w*: {f_star}")

    # Save the chosen solution
    df_star = pd.DataFrame([w_star], columns=['w_MO', 'w_MDF', 'w_MSCM'])
    df_star.to_csv('w_star_topsis.csv', index=False)

    # VISUALIZATION
    print("Generating Visualizations...")
    
   
    try:
        conv_df = pd.read_csv('convergence_history.csv')
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for r in conv_df['run'].unique():
            run_data = conv_df[conv_df['run'] == r]
            axs[0].plot(run_data['gen'], run_data['f1_min'], label=f'Run {r}', alpha=0.7)
            axs[1].plot(run_data['gen'], run_data['f2_min'], label=f'Run {r}', alpha=0.7)
            axs[2].plot(run_data['gen'], run_data['f3_min'], label=f'Run {r}', alpha=0.7)
        
        axs[0].set_title('Convergence of RMSE (f1)')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Minimum RMSE')
        
        axs[1].set_title('Convergence of |1-DIR| (f2)')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Minimum |1-DIR|')
        
        axs[2].set_title('Convergence of Lip95 (f3)')
        axs[2].set_xlabel('Generation')
        axs[2].set_ylabel('Minimum Lip95')
        
        axs[2].legend()
        plt.tight_layout()
        plt.savefig('convergence_plot_v2.png')
        plt.close()
        print("Convergence plot saved to convergence_plot_v2.png")
    except FileNotFoundError:
        print("convergence_history.csv not found, skipping convergence plot.")
    
    # Titik Ideal (Utopian Point)
    f_ideal = np.min(F, axis=0)
    print(f"Ideal Point (Utopian): {f_ideal}")

    # 1. 3D Pareto Front
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(F[:, 0], F[:, 1], F[:, 2], c=F[:, 2], cmap='viridis', alpha=0.8, label='Pareto Surface')
    fig.colorbar(sc, ax=ax, label='Lip95 (Individual Fairness)', shrink=0.5, pad=0.15)
    ax.scatter([f_ideal[0]], [f_ideal[1]], [f_ideal[2]], color='red', marker='*', s=300, label='Ideal Point')
    ax.set_xlabel('RMSE (Accuracy)')
    ax.set_ylabel('|1 - DIR| (Group Fairness)')
    ax.set_zlabel('Lip95 (Individual Fairness)')
    plt.legend()
    plt.title(f'3D Pareto Front\n(Total Solusi Optimal: {len(F)})')
    # Default angle
    ax.view_init(elev=30, azim=-60)
    plt.savefig('pareto_3d_v2.png')

    # Alternate angle to show the spread surface
    ax.view_init(elev=25, azim=135)
    plt.savefig('pareto_3d_alt_v2.png')
    plt.close()

    # 2. 2D Subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: f1 vs f3 (Akurasi vs Individual Fairness)
    axs[0].scatter(F[:, 0], F[:, 2], alpha=0.8, s=30)
    axs[0].scatter([f_ideal[0]], [f_ideal[2]], color='red', marker='*', s=200, label='Ideal Point')
    axs[0].set_title('Akurasi vs Individual Fairness')
    axs[0].set_xlabel('f1 (RMSE) - Lower is Better')
    axs[0].set_ylabel('f3 (Lipschitz Q95) - Lower is Better')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: f3 vs f2 (Individual vs Group Fairness)
    axs[1].scatter(F[:, 2], F[:, 1], alpha=0.8, s=30)
    axs[1].scatter([f_ideal[2]], [f_ideal[1]], color='red', marker='*', s=200)
    axs[1].set_title('Individual vs Group Fairness')
    axs[1].set_xlabel('f3 (Lipschitz Q95) - Lower is Better')
    axs[1].set_ylabel('f2 (|1 - DIR|) - Lower is Better')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: f2 vs f1 (Group Fairness vs Akurasi)
    axs[2].scatter(F[:, 1], F[:, 0], alpha=0.8, s=30)
    axs[2].scatter([f_ideal[1]], [f_ideal[0]], color='red', marker='*', s=200)
    axs[2].set_title('Group Fairness vs Akurasi')
    axs[2].set_xlabel('f2 (|1 - DIR|) - Lower is Better')
    axs[2].set_ylabel('f1 (RMSE) - Lower is Better')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('pareto_2d_subplots_v2.png')
    plt.close()

    # 3. Barycentric Simplex Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    # Coordinates conversion for w = (w1, w2, w3)
    # MO = (0, 0), MDF = (1, 0), MSCM = (0.5, sqrt(3)/2)
    x = weights[:, 1] + 0.5 * weights[:, 2]
    y = weights[:, 2] * (np.sqrt(3) / 2)

    ax.scatter(x, y, alpha=0.5, label='Pareto Solutions')

    # Triangle vertices
    vertices = np.array([[0,0], [1,0], [0.5, np.sqrt(3)/2], [0,0]])
    ax.plot(vertices[:, 0], vertices[:, 1], 'k-')

    ax.text(vertices[0, 0] - 0.05, vertices[0, 1] - 0.05, 'MO (1, 0, 0)', fontsize=12, ha='center')
    ax.text(vertices[1, 0] + 0.05, vertices[1, 1] - 0.05, 'MDF (0, 1, 0)', fontsize=12, ha='center')
    ax.text(vertices[2, 0], vertices[2, 1] + 0.05, 'MSCM (0, 0, 1)', fontsize=12, ha='center', fontweight='bold')

    ax.axis('off')
    ax.set_aspect('equal')
    plt.legend()
    plt.title('Barycentric Simplex of Ensemble Weights')
    plt.savefig('simplex_plot_v2.png')
    plt.close()

    print("Plots saved: convergence_plot_v2.png, pareto_3d_v2.png, pareto_3d_alt_v2.png, pareto_2d_subplots_v2.png, simplex_plot_v2.png")

if __name__ == "__main__":
    main()
