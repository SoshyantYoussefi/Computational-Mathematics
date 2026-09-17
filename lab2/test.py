import matplotlib.pyplot as plt
import numpy as np
import time


ns = np.arange(500, 8001, 500)
ts = np.zeros_like(ns, dtype=float)
for j in range(len(ns)):
    n = ns[j]
    A = np.random.randn(n, n) # create a matrix with random number entries
    x = np.random.randn(n, 1)
    start = time.time() # start the timer
    for _ in range(15): # repeat computation fifteen times
        A @ x

    end = time.time() # read the timer
    ts[j] = (end - start) / 15 # average time per run


plt.loglog(ns, ts, '.-r', markersize=10, linewidth=1.5, label='data') # Plot the data
plt.xlabel('matrix size (n)', fontsize=20)
plt.ylabel('time (sec)', fontsize=20)
ref_line = ts[-1] * (ns / ns[-1])**2
plt.loglog(ns, ref_line, '--k', linewidth=1.5, label='O(n^2)') # Plot a reference line
plt.tick_params(labelsize=20)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right', fontsize=16) # Add a legend
plt.show()