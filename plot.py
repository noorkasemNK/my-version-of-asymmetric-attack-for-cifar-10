import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
import os

plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid errors if not installed
sns.set_theme()

TOTAL_COST = 10000
QUERY_COST = 1
X = np.arange(TOTAL_COST)

# Load log files if they exist
cgba_result = None
hsja_result = None
opt_result = None
surfree_result = None
geoda_result = None

cgba_file = f"attack_CGBA_{TOTAL_COST}.0_1.0_None_0.0.log"
hsja_file = f"attack_HSJA_{TOTAL_COST}.0_1.0_None_0.0.log"
opt_file = f"attack_OPT_{TOTAL_COST}.0_1.0_None_0.0.log"
surfree_file = f"attack_SURFREE_{TOTAL_COST}.0_1.0_None_0.0.log"
geoda_file = f"attack_GEODA_{TOTAL_COST}.0_1.0_None_0.0.log"

if os.path.exists(cgba_file):
    with open(cgba_file, "rb") as fp:
        cgba_result = pickle.load(fp)
else:
    print(f"Warning: {cgba_file} not found")

if os.path.exists(hsja_file):
    with open(hsja_file, "rb") as fp:
        hsja_result = pickle.load(fp)
else:
    print(f"Warning: {hsja_file} not found")

if os.path.exists(opt_file):
    with open(opt_file, "rb") as fp:
        opt_result = pickle.load(fp)

if os.path.exists(surfree_file):
    with open(surfree_file, "rb") as fp:
        surfree_result = pickle.load(fp)
else:
    print(f"Warning: {surfree_file} not found")

if os.path.exists(geoda_file):
    with open(geoda_file, "rb") as fp:
        geoda_result = pickle.load(fp)
else:
    print(f"Warning: {geoda_file} not found")

def get_per_iteration(logs: dict):
    costs = []
    norms = []
    for log in logs:
        for k, v in log.items():
            if len(k.split(',')) == 2 and k.split(',')[0] == 'norm':
                norms.append(v)
            if len(k.split(',')) == 2 and k.split(',')[0] == 'cost':
                costs.append(v)
    # Ensure we have matching lengths - prepend 0 to costs if needed
    if len(costs) > 0 and len(norms) == len(costs) + 1:
        costs = [0] + costs
    elif len(costs) == 0:
        # No costs, return zeros
        return np.zeros_like(X)
    # If lengths still don't match, trim the longer one
    min_len = min(len(costs), len(norms))
    costs = costs[:min_len]
    norms = norms[:min_len]
    if len(costs) == 0 or len(norms) == 0:
        return np.zeros_like(X)
    f = interp1d(costs, norms, bounds_error=False, fill_value=(norms[0], norms[-1]))
    return f(X)

def get_attack_plot(logs):
    ans = []
    for log in logs:
        ans.append(get_per_iteration(log[1]))
    return np.median(ans, axis=0)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot available attack results
# plt.plot(X, np.log(get_attack_plot(cgba_result)), linestyle='-', label='CGBA')
# plt.plot(X, np.log(get_attack_plot(hsja_result)), linestyle='-.', label='HSJA')
# plt.plot(X, np.log(get_attack_plot(surfree_result)), linestyle='--', label='SURFREE')
# plt.plot(X, np.log(get_attack_plot(geoda_result)), linestyle='--', label='GEODA', dashes=(10, 5, 20, 5))
# plt.plot(X, np.log(get_attack_plot(opt_result)), linestyle=':', label='OPT')

has_any_data = False
if cgba_result:
    ax.plot(X, get_attack_plot(cgba_result), linestyle='-', label='CGBA')
    has_any_data = True
if hsja_result:
    ax.plot(X, get_attack_plot(hsja_result), linestyle='-.', label='HSJA')
    has_any_data = True
if surfree_result:
    ax.plot(X, get_attack_plot(surfree_result), linestyle='--', label='SURFREE')
    has_any_data = True
if geoda_result:
    ax.plot(X, get_attack_plot(geoda_result), linestyle='--', label='GEODA', dashes=(10, 5, 20, 5))
    has_any_data = True
if opt_result:
    ax.plot(X, get_attack_plot(opt_result), linestyle=':', label='OPT')
    has_any_data = True

if not has_any_data:
    print("ERROR: No log files found. Please run main.py first to generate attack results.")
    print("Expected log files:")
    print(f"  - {cgba_file}")
    print(f"  - {hsja_file}")
    print(f"  - {surfree_file}")
    print(f"  - {geoda_file}")
    exit(1)

ax.set_title(r"$C_{total}=10000$ $C_{flagged}=1$")
ax.set_xlabel('Cost')
ax.set_ylabel(r'Median $l_2$ perturbation')
ax.legend()
# plt.ylim(0, 50)

# Save plot to file
output_file = "attack_results_plot.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Also save as PDF
output_pdf = "attack_results_plot.pdf"
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Plot saved to: {output_pdf}")

# Don't show plot in non-interactive mode
# plt.show()

# Print table results
print("\n" + "="*50)
print("TABLE RESULTS:")
print("="*50)
if cgba_result:
    print("TABLE RESULTS: CGBA", get_attack_plot(cgba_result)[1000])
if opt_result:
    print("TABLE RESULTS: OPT", get_attack_plot(opt_result)[-1])
if hsja_result:
    print("TABLE RESULTS: HSJA", get_attack_plot(hsja_result)[1000])
if surfree_result:
    print("TABLE RESULTS: SURFREE", get_attack_plot(surfree_result)[1000])
if geoda_result:
    print("TABLE RESULTS: GEODA", get_attack_plot(geoda_result)[1000])