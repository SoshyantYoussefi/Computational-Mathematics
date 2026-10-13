import numpy as np  
from thomas import thomas


def solve_bvp(n):
    h = 1.0 / n
    x = np.linspace(1.0, 2.0, n + 1)
    xi = x[1:-1]

    # Koefficienterna från diskretiseringen
    a = 1 / h**2 - 2 / (xi * h)
    d = -2 / h**2 + 2 / xi**2
    c = 1 / h**2 + 2 / (xi * h)
    b = 2 * np.log(xi) / xi**2

    # Ta hänsyn till randvillkoren
    b[0] -= a[0] * 0.5
    b[-1] -= c[-1] * np.log(2)

    # Lös de n-1 obekanta med Thomasalgoritmen
    y_inner = thomas(a[1:], d, c[:-1], b)

    # Lägg till de kända randvärdena
    y = np.concatenate(([0.5], y_inner, [np.log(2)]))

    return x, y


def exact(x):
    return 4 / x - 2 / x**2 + np.log(x) - 1.5



# Task 1: Solve the tridiagonal system
print("Task 1: Solution using Thomas algorithm")

n = 10
x, y = solve_bvp(n)

for i in range(len(x)):
    print(f"x = {x[i]:.2f}, y = {y[i]:.8f}")


# Task 2: Compute errors and verify second-order accuracy
print("\nTask 2: Error analysis")

errors = []

for n in [10, 20, 40, 80]:
    x, y = solve_bvp(n)
    error = np.max(np.abs(y - exact(x)))
    errors.append(error)

for i, n in enumerate([10, 20, 40, 80]):
    if i < len(errors) - 1:
        ratio = errors[i] / errors[i + 1]
        print(f"n={n}, error={errors[i]:.8e}, ratio={ratio:.4f}")
    else:
        print(f"n={n}, error={errors[i]:.8e}, ratio=--")

print("\nConclusion: The method is second-order accurate.")