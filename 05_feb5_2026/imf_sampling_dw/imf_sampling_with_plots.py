import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# --- Configuration ---
TRUE_ALPHA = 2.35
N_TRIALS = 500  
# Plot 1 Settings
N_STARS_LIST = [10, 30, 100, 300, 1000, 3000, 10000]
SCENARIOS_PLOT1 = [
    {"label": "Broad (1 - 100 M$_{\odot}$, R=100)", "m_min": 1., "m_max": 100.0, "color": "blue"},
    {"label": "Narrow (20 - 100 M$_{\odot}$, R=5)", "m_min": 20.0, "m_max": 100.0, "color": "red"}
]
# Plot 2 Settings
LOG_DYN_RANGE_VALUES = np.linspace(0.5, 3.0, 15) 
N_VALUES_PLOT2 = [50, 200, 1000, 5000] 
COLORS_PLOT2 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

def generate_imf_masses(n, alpha, m_min, m_max):
    u = np.random.random(n)
    term1 = m_max**(1 - alpha)
    term2 = m_min**(1 - alpha)
    masses = ((term1 - term2) * u + term2) ** (1 / (1 - alpha))
    return masses

def neg_log_likelihood(alpha, masses, m_min, m_max):
    if alpha == 1.0: return np.inf
    n = len(masses)
    integral = (m_max**(1 - alpha) - m_min**(1 - alpha)) / (1 - alpha)
    norm_const = 1.0 / integral
    log_l = n * np.log(norm_const) - alpha * np.sum(np.log(masses))
    return -log_l

def parse_literature_data(filename):
    data_points = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if not line.strip() or not line[0].isdigit(): continue
            tokens = line.split()
            # Find log(N) column
            log_n_idx = -1
            for i, token in enumerate(tokens):
                if '.' in token:
                    try: float(token); log_n_idx = i; break
                    except: continue
            if log_n_idx != -1 and log_n_idx + 6 < len(tokens):
                try:
                    data_points.append({
                        'n': 10**float(tokens[log_n_idx]),
                        'sigma': float(tokens[log_n_idx + 3]),
                        'log_dyn_range': np.log10(float(tokens[log_n_idx + 5])/float(tokens[log_n_idx + 4]))
                    })
                except: continue
    except FileNotFoundError: pass
    return data_points

def run_simulation_and_plot():
    lit_data = parse_literature_data('weisz2013_tab1.txt')
    lit_n = [d['n'] for d in lit_data]
    lit_sigma = [d['sigma'] for d in lit_data]
    lit_log_r = [d['log_dyn_range'] for d in lit_data]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- PLOT 1: Sigma vs N ---
    ax1 = axes[0]
    for scenario in SCENARIOS_PLOT1:
        results_std = []
        for n_stars in N_STARS_LIST:
            slopes = []
            for _ in range(N_TRIALS):
                masses = generate_imf_masses(n_stars, TRUE_ALPHA, scenario['m_min'], scenario['m_max'])
                res = minimize_scalar(neg_log_likelihood, bounds=(0.1, 5.0), 
                                      args=(masses, scenario['m_min'], scenario['m_max']), method='bounded')
                slopes.append(res.x)
            results_std.append(np.std(slopes))
        ax1.loglog(N_STARS_LIST, results_std, linestyle='-', linewidth=2, 
                   color=scenario['color'], label=f"Sim: {scenario['label']}")
    
    if lit_data:
        sc1 = ax1.scatter(lit_n, lit_sigma, c=lit_log_r, cmap='viridis', edgecolors='k', s=50, zorder=10)
        cbar1 = plt.colorbar(sc1, ax=ax1, label=r'Dynamic Range $\log(m_{max}/m_{min})$')
    
    ax1.set_xlabel('Number of Stars (N)')
    ax1.set_ylabel(r'Uncertainty $\sigma_{\alpha}$')
    ax1.set_title('Precision vs Sample Size')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # --- PLOT 2: Sigma vs Dynamic Range ---
    ax2 = axes[1]
    m_max_fixed = 100.0
    for i, n_stars in enumerate(N_VALUES_PLOT2):
        results_std = []
        for log_r in LOG_DYN_RANGE_VALUES:
            m_min_var = m_max_fixed / (10**log_r)
            slopes = []
            for _ in range(N_TRIALS):
                masses = generate_imf_masses(n_stars, TRUE_ALPHA, m_min_var, m_max_fixed)
                res = minimize_scalar(neg_log_likelihood, bounds=(0.1, 5.0), 
                                      args=(masses, m_min_var, m_max_fixed), method='bounded')
                slopes.append(res.x)
            results_std.append(np.std(slopes))
        ax2.plot(LOG_DYN_RANGE_VALUES, results_std, linestyle='-', linewidth=2, 
                 color=COLORS_PLOT2[i], label=f"N = {n_stars}")

    if lit_data:
        sc2 = ax2.scatter(lit_log_r, lit_sigma, c=np.log10(lit_n), cmap='plasma', edgecolors='k', s=50, zorder=10)
        cbar2 = plt.colorbar(sc2, ax=ax2, label=r'$\log_{10}(N)$')

    ax2.set_xlabel(r'Dynamic Range $\log(m_{max}/m_{min})$')
    ax2.set_ylabel(r'Uncertainty $\sigma_{\alpha}$')
    ax2.set_title('Precision vs Dynamic Range')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation_and_plot()
