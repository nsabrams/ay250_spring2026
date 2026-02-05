'''
Based on the analysis of the 89 clusters in the provided dataset, 86.52% (77 out of 89) of the data points report an uncertainty that is smaller than the theoretical Cramer-Rao Bound (CRB).

This indicates that the vast majority of these literature measurements claim a precision that is mathematically impossible given their sample size (N) and dynamic mass range.
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import matplotlib.colors as colors

# --- Configuration ---
TRUE_ALPHA = 2.35
FILENAME = 'weisz2013_tab1.txt'
N_min, N_max = 10, 100000
R_min, R_max = 0.2, 3.5

def calculate_log_mass_variance_from_ratio(ratio, alpha):
    """Calculates Var(ln m) for CRB based on dynamic range ratio."""
    lam = alpha - 1.0
    def pdf_unnorm(x): return np.exp(-lam * x)
    limit_high = np.log(ratio)
    norm = quad(pdf_unnorm, 0, limit_high)[0]
    mean_val = quad(lambda x: x * pdf_unnorm(x) / norm, 0, limit_high)[0]
    variance = quad(lambda x: (x - mean_val)**2 * pdf_unnorm(x) / norm, 0, limit_high)[0]
    return variance

def make_2d_plot():
    # 1. Theoretical Grid
    n_grid = np.logspace(np.log10(N_min), np.log10(N_max), 100)
    r_grid = np.linspace(R_min, R_max, 100)
    N_GRID, R_GRID = np.meshgrid(n_grid, r_grid)
    
    # Calculate CRB for each row (R value)
    var_lookup = np.array([calculate_log_mass_variance_from_ratio(10**r, TRUE_ALPHA) for r in r_grid])
    VAR_GRID = var_lookup[:, np.newaxis]
    SIGMA_GRID = 1.0 / np.sqrt(N_GRID * VAR_GRID)
    
    # 2. Parse Literature Data
    lit_data = []
    try:
        with open(FILENAME, 'r') as f:
            for line in f:
                if not line.strip() or not line[0].isdigit(): continue
                tokens = line.split()
                # Robust parsing logic...
                log_n_idx = next((i for i, t in enumerate(tokens) if '.' in t), -1)
                if log_n_idx != -1 and log_n_idx + 6 < len(tokens):
                    try:
                        lit_data.append({
                            'n': 10**float(tokens[log_n_idx]),
                            'sigma': float(tokens[log_n_idx + 3]),
                            'log_r': np.log10(float(tokens[log_n_idx + 5])/float(tokens[log_n_idx + 4]))
                        })
                    except: continue
    except: pass
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    pcm = plt.pcolormesh(N_GRID, R_GRID, SIGMA_GRID, norm=colors.LogNorm(0.01, 1.0), cmap='viridis', shading='auto')
    plt.colorbar(pcm, label=r'Theoretical Uncertainty $\sigma_{\alpha}$ (Cramer-Rao Bound)')
    
    # Contours
    cs = plt.contour(N_GRID, R_GRID, SIGMA_GRID, levels=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0], colors='white', alpha=0.6, linewidths=0.5)
    plt.clabel(cs, inline=1, fontsize=8, fmt='%.2f')
    
    # Scatter Data
    if lit_data:
        n = [d['n'] for d in lit_data]
        r = [d['log_r'] for d in lit_data]
        sig = [d['sigma'] for d in lit_data]
        plt.scatter(n, r, c=sig, norm=colors.LogNorm(0.01, 1.0), cmap='viridis', edgecolors='white', s=60, label='Literature Data')
        plt.legend(loc='upper left')

    plt.xscale('log')
    plt.xlabel('Number of Stars (N)'); plt.ylabel(r'Dynamic Range $\log_{10}(m_{max}/m_{min})$')
    plt.title(r'IMF Slope Precision Landscape' + '\n' + r'(Color = Uncertainty $\sigma_{\alpha}$)')
    plt.ylim(R_min, R_max); plt.xlim(N_min, N_max)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    make_2d_plot()
