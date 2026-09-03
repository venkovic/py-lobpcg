import os
import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import lobpcg, LinearOperator
import time

from Lobpcg import LOBPCG
from BjPreconditioner import assembleBj

# Set random seed for reproducibility
np.random.seed(1)

# Matrix and preconditioner selection
matrix = "bcsst15"

# SPD-SPD generalized: 
# "bcsst12", "bcsst25", "Poisson4", "Poisson8",
# "Poisson16", "Poisson32", "Poisson64", "Poisson128"
#
# SPD standard:
# "bcsst15", "bcsst21", "bodyy6", "tmt_sym",
# "Poisson4_k", "Poisson8_k", "Poisson16_k", 
# "Poisson32_k", "Poisson64_k", "Poisson128_k"


# Results for
# (m, nev) = (200, 10), tol = 1e-5, A_products = :implicit, precond = "bJ10"
#
# "bcsst15" fails with :Basic @ it. 2
# "bcsst15" fails with :BLOPEX @ it. 3
# "bcsst15" converges with :Ortho @ it. 64
# "bcsst15" converges with :Skip_ortho @ it. 64
# "bcsst15" does not converge with :SciPy
#
# "bcsst21" fails with :Basic @ it. 38
# "bcsst21" fails with :BLOPEX @ it. 45
# "bcsst21" converges with :Ortho @ it. 52
# "bcsst21" converges with :Skip_ortho @ it. 52
# "bcsst21" coverges with :SciPy @ it. ~55
#
# "bodyy6" fails with :Basic @ it. 154
# "bodyy6" fails with :BLOPEX @ it. 120
# "bodyy6" converges with :Ortho @ it. 1_285
# "bodyy6" converges with :Skip_ortho @ it. 1_256
# "bodyy6" does not coverge with :SciPy
#
# "tmt_sym" fails with :Basic @ it. 45
# "tmt_sym" converges with :BLOPEX @ it. 42
# "tmt_sym" converges with :Ortho @ it. 58
# "tmt_sym" converges with :Skip_ortho @ it. 58
# "tmt_sym" does not converge with :SciPy 


precond = "bJ10" # None, "bJ#"
matrix_source = "../matrix-market/"
m = 200
nev = 10
tol = 1e-5
num_threads = 6 # Number of threads to use for BLAS operations

# Set BLAS to use the specified number of threads
os.environ['OMP_NUM_THREADS'] = str(num_threads)


# SPD-SPD generalized problems
B = None
if matrix == "bcsst12": # structural mechanics, Harwell-Boeing
  A = mmread(matrix_source + "bcsstk12.mtx").tocsc() # n = 3,948 | nnz = 117,816
  B = mmread(matrix_source + "bcsstm12.mtx").tocsc() # n = 3,948 | nnz = 117,816
if matrix == "bcsst25": # structural mechanics, Harwell-Boeing
  A = mmread(matrix_source + "bcsstk25.mtx").tocsc() # n = 3,948 | nnz = 117,816
  B = mmread(matrix_source + "bcsstm25.mtx").tocsc() # n = 3,948 | nnz = 117,816
if ("Poisson" in matrix) & ("_k" not in matrix):
  nkDoFs = int(matrix[7:])
  if nkDoFs in (4, 8, 16, 32, 64, 128):
    A = mmread(matrix_source + "Poisson_SExp_sig21.0_L0.1_DoF%d000_K.mtx" % nkDoFs).tocsc()
    B = mmread(matrix_source + "Poisson_SExp_sig21.0_L0.1_DoF%d000_M.mtx" % nkDoFs).tocsc()

# SPD standard problems
elif matrix == "bcsst15": # structural mechanics, Harwell-Boeing
  A = mmread(matrix_source + "bcsstk15.mtx").tocsc() # n = 3,948 | nnz = 117,816
elif matrix == "bcsst21": # structural mechanics, Harwell-Boeing
  A = mmread(matrix_source + "bcsstk21.mtx").tocsc() # n = 3,600 | nnz = 26,600
elif matrix == "bodyy6": # structural mechanics, NASA
  A = mmread(matrix_source + "bodyy6.mtx").tocsc() # n = 19,366 | nnz = 134,208
elif matrix == "tmt_sym": # electromagnetics, CEMW
  A = mmread(matrix_source + "tmt_sym.mtx").tocsc() # n = 726,713 | nnz = 5,080,961    
elif ("Poisson" in matrix) & ("_k" in matrix):
  nkDoFs = int(matrix[7:-2])
  if nkDoFs in (4, 8, 16, 32, 64, 128):
    A = mmread(matrix_source + "Poisson_SExp_sig21.0_L0.1_DoF%d000_K.mtx" % nkDoFs).tocsc()

n = A.shape[0]

# Set-up preconditioner
if precond is None:
  M = None
elif "bJ" in precond:
  nbJ = int(precond[2:])
  bJ = assembleBj(nbJ, A)
  M = LinearOperator((n, n), matvec=bJ.invT)

# Random initial guess
X0 = np.random.rand(n, m)
#X0 = np.load("X0.%s.m%d.npz" % (matrix[:-2], m))


# LOBPCG solver parameters
itmax = 10_000
method = "BLOPEX" # "Basic", "BLOPEX", "Ortho", "Skip_ortho", "SciPy"
A_products = "implicit" # "implicit", "explicit"
B_products = "implicit" # "implicit", "explicit"

if method == "SciPy":
  _itmax = 140
  verb=0
  start_time = time.time()
  if B is None:
    res = lobpcg(A, X0, M=M, largest=False, tol=tol, maxiter=_itmax, verbosityLevel=verb)
  else:
    res = lobpcg(A, X0, B=B, M=M, largest=False, tol=tol, maxiter=_itmax, verbosityLevel=verb)
  print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
else:
  start_time = time.time()
  Lambda, X, res = LOBPCG(A, X0, nev,
                          B=B, T=M,
                          itmax=itmax,
                          method=method,
                          tol=tol,
                          A_products=A_products,
                          B_products=B_products)
  print(f"Elapsed time: {time.time() - start_time:.2f} seconds")