import numpy as np
from newtonRaphson import steffensen

# Define f(x) and its derivative
f = lambda x: 3 * x - np.cos(2 * np.pi * x)

# Problem data
x0 = 0.4
tol = 1e-9
x_exact = 1 / 6

# Run Newton-Raphson
xVals, iterations = steffensen(f, x0, tol)

# Compute values for the table
fVals = f(xVals)
errors = np.abs(x_exact - xVals)
log_errors = np.log(errors)

# Experimental Order of Convergence (EOC)
# According to the formula in the assignment:
# EOC_k = log(|x* - x_k|) / log(|x* - x_{k-1}|)
eoc = np.full(len(xVals), np.nan)

for k in range(1, len(xVals)):
    eoc[k] = log_errors[k] / log_errors[k - 1]

# Print results
print(f"Number of values returned: {iterations}")
print()
print(
    f"{'k':>2} "
    f"{'x_k':>16} "
    f"{'f(x_k)':>16} "
    f"{'error':>16} "
    f"{'log(error)':>16} "
    f"{'EOC':>12}"
)
print("-" * 86)

for k in range(len(xVals)):
    eoc_text = "---" if np.isnan(eoc[k]) else f"{eoc[k]:.6f}"

    print(
        f"{k+1:2d} "
        f"{xVals[k]:16.10e} "
        f"{fVals[k]:16.10e} "
        f"{errors[k]:16.10e} "
        f"{log_errors[k]:16.10e} "
        f"{eoc_text:>12}"
    )

# Average EOC, ignoring the first undefined value
average_eoc = np.nanmean(eoc)

print(f"Average EOC = {average_eoc:.6f}")
