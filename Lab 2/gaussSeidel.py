import numpy as np

def gaussSeidel(A, b, x0, tol):
#
#  Gauss-Seidel iterative method to approximate the solution of a
#  linear system Ax=b up to a user defined tolerance
#
#  INPUT:
#    A   - n by n square, non-singular matrix
#    b   - n by 1 right hand side vector
#    x0  - n by 1 vector containing that initial guess for the iteration
#    tol - user set tolerance for the stopping condition in the iteration
#
#  OUTPUT:
#    x - n by 1 vector containing the iterative solution
#    k - number of iterations

#  get the system size
   n, _ = A.shape

#  Gauss-Seidel iteration which overwrites the current approximate solution
#  with the new approximate solution (pseudocode available in the lecture
#  notes on page 46)
   max_i=50
   for k in range(1, max_i):
      for i in range(1, n):
         sum=0
         for j in range(1, n):
            if j!=i:
               sum=sum+A[i][j]*x0[j]
         x0[i]=(b[i]-sum)/A[i][i]
      r=b-A*x0
      if np.linalg.norm(r)<tol*np.linalg.norm(b):
         break

   return x0, k


n = 10
main_diag = 2.1 * np.ones(n)
off_diag = -1 * np.ones(n - 1)
C = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

b = C @ (3.5 * np.ones(n))
x0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
tol = 1e-8

print(gaussSeidel(C, b, x0, tol ))