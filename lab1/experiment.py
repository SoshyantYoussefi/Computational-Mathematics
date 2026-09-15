import numpy as np

np.set_printoptions(precision=16, suppress=False)

b_hat = np.array([0.2539949, 0.1272061])
b_tilde = np.array([0.252428, 0.126213])


A = np.array([
    [0.913, 0.659],
    [0.457, 0.330]
])

b = np.array([0.254, 0.127])

x_hat = np.linalg.solve(A, b_hat)
x_tilde = np.linalg.solve(A, b_tilde)

#print(abs(1 - x_hat))
#print(abs(1 - x_tilde))

#print("x_hat = ", x_hat)
#print("x_tilde = ", x_tilde)


r_hat = b - A @ x_hat
r_tilde = b - A @ x_tilde

#print("r_hat = " ,r_hat)
#print("r_tilde = ", r_tilde)

ans = np.linalg.norm(r_hat, ord=1)
ans2 = np.linalg.norm(r_tilde, ord=1)

#print(ans)
#print(ans2)

sista = np.linalg.cond(A, p=1)
print(sista)



