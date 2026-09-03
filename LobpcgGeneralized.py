from scipy.linalg.blas import dgemm
from scipy.linalg import cholesky, inv
from numpy import matmul
from numpy.linalg import norm, solve
from numpy import copyto, diag, empty, vstack, zeros
from numpy import abs, sqrt

from RayleighRitz import *
from Ortho import *

def Basic_LOBPCG_gen(A, B, X0, nev,
                     T=None, itmax=200, tol=1e-6,
                     A_products='implicit',
                     B_products='implicit'):
  """
  Knyazev, A. V. (2001)
  Toward the optimal preconditioned eigensolver: Locally optimal block preconditioned conjugate gradient method
  SIAM journal on scientific computing, 23(2), 517-541
  
  Parameters:
  A          : left  hand-side operator, symmetric positive definite, n-by-n
  B          : right hand-side operator, symmetric positive definite, n-by-n
  X0         : initial iterates, n-by-m (m < n)
  nev        : number of wanted eigenpairs, nev <= m
  T          : precondontioner, symmetric positive definite, n-by-n
  itmax      : maximum number of iterations
  tol        : tolerance used for convergence criterion
  A_products : if :implicit, the matrix products with A are updated implicitly
  B_products : if :implicit, the matrix products with B are updated implicitly
    
  Returns:
  Lambda : last iterates of least dominant eigenvalues, m-by-1
  X      : last iterates of least dominant eigenvectors, n-by-m
  res    : normalized norms of eigenresiduals, m-by-it
  """

  n, m = X0.shape
  
  X = empty((n, m))
  R = empty((n, m))
  Z = empty((n, m))
  P = empty((n, m))
  W = empty((n, m))
  AX = empty((n, m))
  AZ = empty((n, m))
  AP = empty((n, m))
  BX = empty((n, m))
  BZ = empty((n, m))
  BP = empty((n, m))
  
  res = empty((m, itmax + 1))
  k = 0
  
  copyto(X, X0)
  AX[:] = A @ X
  BX[:] = B @ X
  hX, Lambda = RR1(X, AX, BX)
  matmul(X, hX, out=W); copyto(X, W) # X[:] = X @ hX
  if A_products == 'implicit':
    matmul(AX, hX, out=W); copyto(AX, W) # AX[:] = AX @ hX
  else:
    AX[:] = A @ X
  if B_products == 'implicit':
    matmul(BX, hX, out=W); copyto(BX, W) # BX[:] = BX @ hX
  else:
    BX[:] = B @ X

  # R[:] = AX - BX @ diag(Lambda)
  copyto(R, AX)
  R[:] = dgemm(alpha=-1.0, a=BX, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
  res[:, 0] = norm(R, axis=0)
  res[:, 0] /= abs(Lambda)
  for i in range(k, nev):
    if res[i, 0] < tol:
      k += 1
    else:
      break
  print("it = 0, k = %d" % k)
  print("extrema(res) =", res[:, 0].min(), res[:, 0].max())
 
  _AX = AX[:, k:m]
  _R = R[:, k:m]; _W = W[:, k:m]
  _Z = Z[:, k:m]; _AZ = AZ[:, k:m]; _BZ = BZ[:, k:m]
  _P = P[:, k:m]; _AP = AP[:, k:m]; _BP = BP[:, k:m]
  
  if k < nev:
    for j in range(1, itmax + 1):
      print("it = %d, k = %d" % (j, k))
      if T is not None:
        _Z[:] = T(_R) 
      else:
        _Z[:] = _R
      _AZ[:] = A @ _Z
      _BZ[:] = B @ _Z
      if j == 1:
        hX, Lambda = RR2(X, _Z, _AX, _AZ, _BZ)
        hX_X = hX[:m, :]
        hX_Z = hX[m:2*m-k, :]
        _hX_Z = hX_Z[:, k:m]
        _P[:] = matmul(_Z, _hX_Z) # _P[:] = _Z @ _hX_Z
        if A_products == 'implicit':
          _AP[:] = matmul(_AZ, _hX_Z) # _AP[:] = _AZ @ _hX_Z
        else:
          _AP[:] = A @ _P
        if B_products == 'implicit':
          _BP[:] = matmul(_BZ, _hX_Z) # _BP[:] = _BZ @ _hX_Z
        else:
          _BP[:] = B @ _P
      else:
        hX, Lambda = RR3(X, _Z, _P, AX, _AZ, _AP, _BZ, _BP)
        hX_X = hX[:m, :]
        hX_Z = hX[m:2*m-k, :]
        hX_P = hX[2*m-k:3*m-2*k, :]
        _hX_Z = hX_Z[:, k:m]
        _hX_P = hX_P[:, k:m]
        # _P[:] = _Z @ _hX_Z + _P @ _hX_P
        _W[:] = matmul(_P, _hX_P)
        _W[:] = dgemm(alpha=1.0, a=_Z, b=_hX_Z, c=_W, overwrite_c=True, beta=1.0)
        copyto(_P, _W)
        if A_products == 'implicit':
          # _AP[:] = _AZ @ _hX_Z + _AP @ _hX_P
          _W[:] = matmul(_AP, _hX_P)
          _W[:] = dgemm(alpha=1.0, a=_AZ, b=_hX_Z, c=_W, overwrite_c=True, beta=1.0)
          copyto(_AP, _W)
        else:
          _AP[:] = A @ _P
        if B_products == 'implicit':
          # _BP[:] = _BZ @ _hX_Z + _BP @ _hX_P
          _W[:] = matmul(_BP, _hX_P)
          _W[:] = dgemm(alpha=1.0, a=_BZ, b=_hX_Z, c=_W, overwrite_c=True, beta=1.0)
          copyto(_BP, _W)
        else:
          _BP[:] = B @ _P
      if k > 0:
        if j == 1:
          # X[:] = X @ hX_X + _Z @ hX_Z
          matmul(_Z, hX_Z, out=W)
          W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(X, W)
        else:
          # X[:] = X @ hX_X + _Z @ hX_Z + _P @ hX_P
          matmul(_P, hX_P, out=W)
          W[:] = dgemm(alpha=1.0, a=_Z, b=hX_Z, c=W, overwrite_c=True, beta=1.0)
          W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(X, W)
        AX[:] = A @ X
        BX[:] = B @ X
      else:
        # X[:] = P + X @ hX_X
        copyto(W, P)
        W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
        copyto(X, W)
        if A_products == 'implicit':
          # AX[:] = AP + AX @ hX_X
          copyto(W, AP)
          W[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(AX, W)
        else:
          AX[:] = A @ X
        if B_products == 'implicit':
          # BX[:] = BP + BX @ hX_X
          copyto(W, BP)
          W[:] = dgemm(alpha=1.0, a=BX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(BX, W)
        else:
          BX[:] = B @ X

      # R[:] = AX - BX @ diag(Lambda)
      copyto(R, AX)
      R[:] = dgemm(alpha=-1.0, a=BX, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
      res[:, j] = norm(R, axis=0)
      res[:, j] /= abs(Lambda)
      print("extrema(res) =", res[:, j].min(), res[:, j].max())

      for i in range(k, nev):
        if res[i, j] < tol:
          k += 1
        else:
          break

      if k >= nev:
        return Lambda, X, res[:, :j+1]
    
      if m - k < _R.shape[1]:
        _AX = AX[:, k:m]
        _R = R[:, k:m]; _W = W[:, k:m]
        _Z = Z[:, k:m]; _AZ = AZ[:, k:m]; _BZ = BZ[:, k:m]
        _P = P[:, k:m]; _AP = AP[:, k:m]; _BP = BP[:, k:m]

  return Lambda, X, res

def BLOPEX_LOBPCG_gen(A, B, X0, nev,
                      T=None, itmax=200, tol=1e-6,
                      A_products='implicit',
                      B_products='implicit'):
  """
  Knyazev, A. V., Argentati, M. E., Lashuk, I., & Ovtchinnikov, E. E. (2007)
  Block locally optimal preconditioned eigenvalue Xolvers (BLOPEX) in Hypre and PETSc
  SIAM Journal on Scientific Computing, 29(5), 2224-2239.

  Parameters:
  A          : left  hand-side operator, symmetric positive definite, n-by-n
  B          : right hand-side operator, symmetric positive definite, n-by-n
  X0         : initial iterates, n-by-m (m < n)
  nev        : number of wanted eigenpairs, nev <= m
  T          : precondontioner, symmetric positive definite, n-by-n
  itmax      : maximum number of iterations
  tol        : tolerance used for convergence criterion
  A_products : if :implicit, the matrix products with A are updated implicitly
  B_products : if :implicit, the matrix products with B are updated implicitly
    
  Returns:
  Lambda : last iterates of least dominant eigenvalues, m-by-1
  X      : last iterates of least dominant eigenvectors, n-by-m
  res    : normalized norms of eigenresiduals, m-by-it
  """

  n, m = X0.shape
  
  X = empty((n, m))
  R = empty((n, m))
  Z = empty((n, m))
  P = empty((n, m))
  W = empty((n, m))
  AX = empty((n, m))
  AZ = empty((n, m))
  AP = empty((n, m))
  BX = empty((n, m))
  BZ = empty((n, m))
  BP = empty((n, m))

  res = empty((m, itmax+1))
  k = 0

  copyto(X, X0)
  BX[:] = B @ X
  XtBX = matmul(X.T, BX)
  U = cholesky(XtBX, lower=False)
  X[:] = solve(U.T, X.T).T # X[:] = X @ inv(U)
  if B_products == 'implicit':
    BX[:] = solve(U.T, BX.T).T # BX[:] = BX @ inv(U)
  else:
    BX[:] = B @ X
  AX[:] = A @ X
  hX, Lambda = RR6(X, AX)
  matmul(X, hX, out=W); copyto(X, W) # X[:] = X @ hX
  if A_products == 'implicit':
    matmul(AX, hX, out=W); copyto(AX, W)  # AX[:] = AX @ hX
  else:
    AX[:] = A @ X
  if B_products == 'implicit':
    matmul(BX, hX, out=W); copyto(BX, W)  # BX[:] = BX @ hX
  else:
    BX[:] = B @ X  

  # R[:] = AX - BX @ diag(Lambda)
  copyto(R, AX)
  R[:] = dgemm(alpha=-1.0, a=BX, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
  res[:, 0] = norm(R, axis=0)
  res[:, 0] /= abs(Lambda)
  for i in range(k, nev):
    if res[i, 0] < tol:
      k += 1
    else:
      break
  print("it = 0, k = %d" % k)
  print("extrema(res) =", res[:, 0].min(), res[:, 0].max())
    
  if k < nev:
    for j in range(1, itmax + 1):      
      print("it = %d, k = %d" % (j, k))
      if T is not None:
        Z[:] = T(R)
      else:
        Z[:] = R
      BZ[:] = B @ Z
      ZtBZ  = matmul(Z.T, BZ)
      U[:] = cholesky(ZtBZ, lower=False)
      Z[:] = solve(U.T, Z.T).T # Z[:] = Z @ inv(U)
      if B_products == 'implicit':
        BZ[:] = solve(U.T, BZ.T).T  # BZ[:] = BZ @ inv(U)
      else:
        BZ[:] = B @ Z
      AZ[:] = A @ Z
      if j == 1:
        hX, Lambda = RR_BLOPEX1(X, Z, AX, AZ, BZ)
        hX_X = hX[:m, :]
        hX_Z = hX[m:, :]
        matmul(Z, hX_Z, out=P) # P[:] = Z @ hX_Z
        if A_products == 'implicit':
          matmul(AZ, hX_Z, out=AP) # AP[:] = AZ @ hX_Z
        else:
          AP[:] = A @ P
        if B_products == 'implicit':
          matmul(BZ, hX_Z, out=BP) # BP[:] = BZ @ hX_Z
        else:
          BP[:] = B @ P
      else:
        PtBP = matmul(P.T, BP)
        U[:] = cholesky(PtBP, lower=False)
        P[:] = solve(U.T, P.T).T # P[:] = P @ inv(U)
        if A_products == 'implicit':
          AP[:] = solve(U.T, AP.T).T # AP[:] = AP @ inv(U)
        else:
          AP[:] = A @ P
        if B_products == 'implicit':
          BP[:] = solve(U.T, BP.T).T # BP[:] = BP @ inv(U)
        else:
          BP[:] = B @ P
        hX, Lambda = RR_BLOPEX2(X, Z, P, AX, AZ, AP, BZ, BP)
        hX_X = hX[:m, :]
        hX_Z = hX[m:2*m, :]
        hX_P = hX[2*m:, :]
        # P[:] = Z @ hX_Z + P @ hX_P
        matmul(P, hX_P, out=W)
        W[:] = dgemm(alpha=1.0, a=Z, b=hX_Z, c=W, overwrite_c=True, beta=1.0)
        copyto(P, W)      
        if A_products == 'implicit':
          # AP[:] = AZ @ hX_Z + AP @ hX_P
          matmul(AP, hX_P, out=W)
          W[:] = dgemm(alpha=1.0, a=AZ, b=hX_Z, c=W, overwrite_c=True, beta=1.0)
          copyto(AP, W)
        else:
          AP[:] = A @ P
        if B_products == 'implicit':
          # BP[:] = BZ @ hX_Z + BP @ hX_P
          matmul(BP, hX_P, out=W)
          W[:] = dgemm(alpha=1.0, a=BZ, b=hX_Z, c=W, overwrite_c=True, beta=1.0)
          copyto(BP, W)  
        else:
          BP[:] = B @ P
      # X[:] = P + X @ hX_X
      copyto(W, P)
      W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
      copyto(X, W)  
      if A_products == 'implicit':
        # AX[:] = AP + AX @ hX_X
        copyto(W, AP)
        W[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
        copyto(AX, W)  
      else:
        AX[:] = A @ X
      if B_products == 'implicit':
        # BX[:] = BP + BX @ hX_X
        copyto(W, BP)
        W[:] = dgemm(alpha=1.0, a=BX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
        copyto(BX, W)  
      else:
        BX[:] = B @ X

      # R[:] = AX - BX @ diag(Lambda)
      copyto(R, AX)
      R[:] = dgemm(alpha=-1.0, a=BX, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
      res[:, j] = norm(R, axis=0)
      res[:, j] /= abs(Lambda)
      print("extrema(res) =", res[:, j].min(), res[:, j].max())
      
      for i in range(k, nev):
        if res[i, j] < tol:
          k += 1
        else:
          break

      if k >= nev:
        return Lambda, X, res[:, :j+1]
    
    return Lambda, X, res
  
def Ortho_LOBPCG_gen(A, B, X0, nev, 
                     T=None, itmax=200, tol=1e-6, 
                     A_products='implicit', 
                     B_products='implicit'):
  """
  # Hetmaniuk, U., & Lehoucq, R. (2006)
  # Basis selection in LOBPCG
  # Journal of Computational Physics, 218(1), 324-332

  Parameters:
  A          : left  hand-side operator, symmetric positive definite, n-by-n
  B          : right hand-side operator, symmetric positive definite, n-by-n
  X0         : initial iterates, n-by-m (m < n)
  nev        : number of wanted eigenpairs, nev <= m
  T          : precondontioner, symmetric positive definite, n-by-n
  itmax      : maximum number of iterations
  tol        : tolerance used for convergence criterion
  A_products : if :implicit, the matrix products with A are updated implicitly
  B_products : if :implicit, the matrix products with B are updated implicitly
    
  Returns:
  Lambda : last iterates of least dominant eigenvalues, m-by-1
  X      : last iterates of least dominant eigenvectors, n-by-m
  res    : normalized norms of eigenresiduals, m-by-it
  """
  
  n, m = X0.shape

  modified = False # True => Duersch et al. (2018)

  R = empty((n, m))
  XP = empty((n, 2*m))
  Z = empty((n, m))
  Q = empty((n, m))
  W = empty((n, m))
  AX = empty((n, m))
  AZ = empty((n, m))
  AP = empty((n, m))
  BX = empty((n, m))
  BZ = empty((n, m))
  BP = empty((n, m))

  X = XP[:, :m]
  P = XP[:, m:2*m]

  res = empty((m, itmax + 1))
  k = 0

  copyto(X, X0)
  AX[:] = A @ X
  BX[:] = B @ X
  hX, Lambda = RR1(X, AX, BX)
  matmul(X, hX, out=W); copyto(X, W) # X[:] = X @ hX
  if A_products == 'implicit':
    matmul(AX, hX, out=W); copyto(AX, W) # AX[:] = AX @ hX
  else:
    AX[:] = A @ X
  if B_products == 'implicit':
    matmul(BX, hX, out=W); copyto(BX, W) # BX[:] = BX @ hX
  else:
    BX[:] = B @ X

  # R[:] = AX - BX @ diag(Lambda)
  copyto(R, AX)
  R[:] = dgemm(alpha=-1.0, a=BX, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
  res[:, 0] = norm(R, axis=0) 
  res[:, 0] /= abs(Lambda)
  for i in range(k, nev):
    if res[i, 0] < tol:
      k += 1
    else:
      break
  print("it = 0, k = %d" % k)
  print("extrema(res) =", res[:, 0].min(), res[:, 0].max())

  if k < nev:
    for j in range(1, itmax + 1):      
      print("it = %d, k = %d" % (j, k))
      if T is not None:
        Z[:] = T(R) 
      else:
        Z[:] = R
      BZ[:] = B @ Z
      if j == 1:
        norm_BX = norm(BX)
        OrthoB(Z, B, X, BZ, norm_BX)
        AZ[:] = A @ Z
        if modified:
          hX, Lambda, hY = RR4(X, Z, AX, AZ, modified=True)
          hX_X, hX_Z = hX[:m, :], hX[m:2*m, :]
        else:
          hX, Lambda = RR4(X, Z, AX, AZ)
          hX_X, hX_Z = hX[:m, :], hX[m:2*m, :]
          hY = vstack([zeros((m, m)), hX_Z])
          Ortho(hY, vstack([hX_X, hX_Z]))
        hY_X, hY_Z = hY[:m, :], hY[m:2*m, :]
        if A_products == 'implicit':
          # AP[:] = AX @ hY_X + AZ @ hY_Z
          matmul(AZ, hY_Z, out=AP)
          AP[:] = dgemm(alpha=1.0, a=AX, b=hY_X, c=AP, overwrite_c=True, beta=1.0)
          # AX[:] = AX @ hX_X + AZ @ hX_Z
          matmul(AZ, hX_Z, out=W)
          W[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(AX, W)      
        if B_products == 'implicit':
          # BP[:] = BX @ hY_X + BZ @ hY_Z
          matmul(BZ, hY_Z, out=BP)
          BP[:] = dgemm(alpha=1.0, a=BX, b=hY_X, c=BP, overwrite_c=True, beta=1.0)
          # BX[:] = BX @ hX_X + BZ @ hX_Z
          matmul(BZ, hX_Z, out=W)
          W[:] = dgemm(alpha=1.0, a=BX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          copyto(BX, W)   
        # W[:] = X @ hX_X + Z @ hX_Z
        matmul(Z, hX_Z, out=W)
        W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
        # P[:] = X @ hY_X + Z @ hY_Z
        matmul(Z, hY_Z, out=P)
        P[:] = dgemm(alpha=1.0, a=X, b=hY_X, c=P, overwrite_c=True, beta=1.0)
      else:
        norm_BXP = sqrt(norm(BX)**2 + norm(BP)**2)
        OrthoB(Z, B, XP, BZ, norm_BXP)
        AZ[:] = A @ Z
        if modified:
          hX, Lambda, hY = RR5(X, Z, P, AX, AZ, AP, modified=True)
          hX_X, hX_Z, hX_P = hX[:m, :], hX[m:2*m, :], hX[2*m:3*m, :]
        else:
          hX, Lambda = RR5(X, Z, P, AX, AZ, AP)
          hX_X, hX_Z, hX_P = hX[:m, :], hX[m:2*m, :], hX[2*m:3*m, :]
          hY = vstack([zeros((m, m)), hX_Z, hX_P])
          Ortho(hY, vstack([hX_X, hX_Z, hX_P]))
        hY_X, hY_Z, hY_P = hY[:m, :], hY[m:2*m, :], hY[2*m:3*m, :]
        if A_products == 'implicit':
          # W[:] = AX @ hY_X + AZ @ hY_Z + AP @ hY_P
          matmul(AP, hY_P, out=W)
          W[:] = dgemm(alpha=1.0, a=AZ, b=hY_Z, c=W, overwrite_c=True, beta=1.0)
          W[:] = dgemm(alpha=1.0, a=AX, b=hY_X, c=W, overwrite_c=True, beta=1.0)
          # AX[:] = AX @ hX_X + AZ @ hX_Z + AP @ hX_P
          matmul(AP, hX_P, out=Q)
          Q[:] = dgemm(alpha=1.0, a=AZ, b=hX_Z, c=Q, overwrite_c=True, beta=1.0)
          Q[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=Q, overwrite_c=True, beta=1.0)
          copyto(AX, Q)
          # AP[:] = W
          copyto(AP, W)
        if B_products == 'implicit':
          #W[:] = BX @ hY_X + BZ @ hY_Z + BP @ hY_P
          matmul(BP, hY_P, out=W)
          W[:] = dgemm(alpha=1.0, a=BZ, b=hY_Z, c=W, overwrite_c=True, beta=1.0)
          W[:] = dgemm(alpha=1.0, a=BX, b=hY_X, c=W, overwrite_c=True, beta=1.0)
          #BX[:] = BX @ hX_X + BZ @ hX_Z + BP @ hX_P
          matmul(BP, hX_P, out=Q)
          Q[:] = dgemm(alpha=1.0, a=BZ, b=hX_Z, c=Q, overwrite_c=True, beta=1.0)
          Q[:] = dgemm(alpha=1.0, a=BX, b=hX_X, c=Q, overwrite_c=True, beta=1.0)
          copyto(BX, Q)
          # BP[:] = W
          copyto(BP, W)
        # W[:] = X @ hX_X + Z @ hX_Z + P @ hX_P
        matmul(P, hX_P, out=W)
        W[:] = dgemm(alpha=1.0, a=Z, b=hX_Z, c=W, overwrite_c=True, beta=1.0)
        W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
        # Q[:] = X @ hY_X + Z @ hY_Z + P @ hY_P
        matmul(P, hY_P, out=Q)
        Q[:] = dgemm(alpha=1.0, a=Z, b=hY_Z, c=Q, overwrite_c=True, beta=1.0)
        Q[:] = dgemm(alpha=1.0, a=X, b=hY_X, c=Q, overwrite_c=True, beta=1.0)
        # P[:] = Q
        copyto(P, Q)
      copyto(X, W)
      if A_products != 'implicit':
        AX[:] = A @ X
        AP[:] = A @ P
      if B_products != 'implicit':
        BX[:] = B @ X
        BP[:] = B @ P

      # R[:] = AX - BX @ diag(Lambda)
      copyto(R, AX)
      R[:] = dgemm(alpha=-1.0, a=BX, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
      res[:, j] = norm(R, axis=0) / abs(Lambda)
      print("extrema(res) =", res[:, j].min(), res[:, j].max())

      for i in range(k, nev):
        if res[i, j] < tol:
          k += 1
        else:
          break
        
      if k >= nev:
        return Lambda, X, res[:, :j+1]

  return Lambda, X, res

def Skip_ortho_LOBPCG_gen(A, B, X0, nev,
                          T=None, itmax=200, tol=1e-6,
                          A_products='implicit',
                          B_products='implicit'):
  """
  Duersch, J. A., Shao, M., Yang, C., & Gu, M. (2018)
  A robust and efficient implementation of LOBPCG
  SIAM Journal on Scientific Computing, 40(5), C655-C676
  
  Parameters:
  A          : left  hand-side operator, symmetric positive definite, n-by-n
  B          : right hand-side operator, symmetric positive definite, n-by-n
  X0         : initial iterates, n-by-m (m < n)
  nev        : number of wanted eigenpairs, nev <= m
  T          : precondontioner, symmetric positive definite, n-by-n
  itmax      : maximum number of iterations
  tol        : tolerance used for convergence criterion
  A_products : if :implicit, the matrix products with A are updated implicitly
  B_products : if :implicit, the matrix products with B are updated implicitly
    
  Returns:
  Lambda : last iterates of least dominant eigenvalues, m-by-1
  X      : last iterates of least dominant eigenvectors, n-by-m
  res    : normalized norms of eigenresiduals, m-by-it
  """
  
  n, m = X0.shape
  
  skip_ortho = True
  modified = True

  R = empty((n, m))
  XP = empty((n, 2*m))
  Z = empty((n, m))
  W = empty((n, m))
  Q = empty((n, m))
  AX = empty((n, m))
  AZ = empty((n, m))
  AP = empty((n, m))
  BX = empty((n, m))
  BZ = empty((n, m))
  BP = empty((n, m))
    
  X = XP[:, :m]
  P = XP[:, m:2*m]
    
  res = empty((m, itmax + 1))
  k = 0

  copyto(X, X0)
  AX[:] = A @ X
  BX[:] = B @ X
  hX, Lambda = RR1(X, AX, BX)
  matmul(X, hX, out=W); copyto(X, W) # X[:] = X @ hX
  if A_products == 'implicit':
    matmul(AX, hX, out=W); copyto(AX, W) # AX[:] = AX @ hX
  else:
    AX[:] = A @ X
  if B_products == 'implicit':
    matmul(BX, hX, out=W); copyto(BX, W) # BX[:] = BX @ hX
  else:
    BX[:] = B @ X
  
  # R[:] = AX - BX @ diag(Lambda)
  copyto(R, AX)
  R[:] = dgemm(alpha=-1.0, a=BX, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
  res[:, 0] = norm(R, axis=0)
  res[:, 0] /= abs(Lambda)
  for i in range(k, nev):
    if res[i, 0] < tol:
      k += 1
    else:
      break
  print("it = 0, k = %d" % k)
  print("extrema(res) =", res[:, 0].min(), res[:, 0].max())

  if k < nev:
    for j in range(1, itmax + 1):
      print("it = %d, k = %d, skip_ortho = %d" % (j, k, skip_ortho))
      if T is not None:
        Z[:] = T(R) 
      else:
        Z[:] = R
      if j == 1:
        if skip_ortho:
          BZ[:] = B @ Z
          AZ[:] = A @ Z
          hX, Lambda, skip_ortho, VtBV = RR2(X, Z, AX, AZ, BZ, True)
          hX_X, hX_Z = hX[:m, :], hX[m:2*m, :]
          if skip_ortho:
            hY = vstack([zeros((m, m)), hX_Z])
            norm_VtBVXZ = norm(VtBV @ vstack([hX_X, hX_Z]))
            OrthoB(hY, VtBV, vstack([hX_X, hX_Z]), VtBV @ hY, norm_VtBVXZ)
            hY_X, hY_Z = hY[:m, :], hY[m:2*m, :]
            if A_products == 'implicit':
              # AP[:] = AX @ hY_X + AZ @ hY_Z
              matmul(AZ, hY_Z, out=AP)
              AP[:] = dgemm(alpha=1.0, a=AX, b=hY_X, c=AP, overwrite_c=True, beta=1.0)
              # AX[:] = AX @ hX_X + AZ @ hX_Z
              matmul(AZ, hX_Z, out=W)
              W[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
              copyto(AX, W)            
            if B_products == 'implicit':
              # BP[:] = BX @ hY_X + BZ @ hY_Z
              matmul(BZ, hY_Z, out=BP)
              BP[:] = dgemm(alpha=1.0, a=BX, b=hY_X, c=BP, overwrite_c=True, beta=1.0)
              # BX[:] = BX @ hX_X + BZ @ hX_Z
              matmul(BZ, hX_Z, out=W)
              W[:] = dgemm(alpha=1.0, a=BX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
              copyto(BX, W)  
            # W[:] = X @ hX_X + Z @ hX_Z
            matmul(Z, hX_Z, out=W)
            W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
            # P[:] = X @ hY_X + Z @ hY_Z
            matmul(Z, hY_Z, out=P)
            P[:] = dgemm(alpha=1.0, a=X, b=hY_X, c=P, overwrite_c=True, beta=1.0)
            copyto(X, W)
            if A_products != 'implicit':
              AP[:] = A @ P
              AX[:] = A @ X
            if B_products != 'implicit':
              BP[:] = B @ P
              BX[:] = B @ X
        if not skip_ortho:
          BZ[:] = B @ Z
          norm_BX = norm(BX)
          OrthoB(Z, B, X, BZ, norm_BX)
          AZ[:] = A @ Z
          if modified:
            hX, Lambda, hY = RR4(X, Z, AX, AZ, modified=True)
            hX_X, hX_Z = hX[:m, :], hX[m:2*m, :]
          else:
            hX, Lambda = RR4(X, Z, AX, AZ)
            hX_X, hX_Z = hX[:m, :], hX[m:2*m, :]
            hY = vstack([zeros((m, m)), hX_Z])
            Ortho(hY, vstack([hX_X, hX_Z]))       
          hY_X, hY_Z = hY[:m, :], hY[m:2*m, :]
          if A_products == 'implicit':
            # AP[:] = AX @ hY_X + AZ @ hY_Z
            matmul(AZ, hY_Z, out=AP)
            AP[:] = dgemm(alpha=1.0, a=AX, b=hY_X, c=AP, overwrite_c=True, beta=1.0)
            # AX[:] = AX @ hX_X + AZ @ hX_Z
            matmul(AZ, hX_Z, out=W)
            W[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
            copyto(AX, W)     
          else:
            AP[:] = A @ P
            AX[:] = A @ X
          if B_products == 'implicit':
            # BP[:] = BX @ hY_X + BZ @ hY_Z
            matmul(BZ, hY_Z, out=BP)
            BP[:] = dgemm(alpha=1.0, a=BX, b=hY_X, c=BP, overwrite_c=True, beta=1.0)
            # BX[:] = BX @ hX_X + BZ @ hX_Z
            matmul(BZ, hX_Z, out=W)
            W[:] = dgemm(alpha=1.0, a=BX, b=hX_X, c=W, overwrite_c=True, beta=1.0)
            copyto(BX, W)
          else:
            BP[:] = B @ P
            BX[:] = B @ X
          # W[:] = X @ hX_X + Z @ hX_Z
          matmul(Z, hX_Z, out=W)
          W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          # P[:] = X @ hY_X + Z @ hY_Z
          matmul(Z, hY_Z, out=P)
          P[:] = dgemm(alpha=1.0, a=X, b=hY_X, c=P, overwrite_c=True, beta=1.0)
          # X[:] = W
          copyto(X, W)
          if A_products != 'implicit':
            AX[:] = A @ X
            AP[:] = A @ P
          if B_products != 'implicit':
            BX[:] = B @ X
            BP[:] = B @ P
      else:
        if skip_ortho:
          BZ[:] = B @ Z
          AZ[:] = A @ Z
          hX, Lambda, skip_ortho, VtBV = RR3(X, Z, P, AX, AZ, AP, BZ, BP, True)
          hX_X, hX_Z, hX_P = hX[:m, :], hX[m:2*m, :], hX[2*m:3*m, :]
          if skip_ortho:
            hY = vstack([zeros((m, m)), hX_Z, hX_P])
            norm_VtBVXZP = norm(VtBV @ vstack([hX_X, hX_Z, hX_P]))
            OrthoB(hY, VtBV, vstack([hX_X, hX_Z, hX_P]), VtBV @ hY, norm_VtBVXZP)
            hY_X, hY_Z, hY_P = hY[:m, :], hY[m:2*m, :], hY[2*m:3*m, :]
            if A_products == 'implicit':
              # W[:] = AX @ hY_X + AZ @ hY_Z + AP @ hY_P
              matmul(AP, hY_P, out=W)
              W[:] = dgemm(alpha=1.0, a=AZ, b=hY_Z, c=W, overwrite_c=True, beta=1.0)
              W[:] = dgemm(alpha=1.0, a=AX, b=hY_X, c=W, overwrite_c=True, beta=1.0)
              # AX[:] = AX @ hX_X + AZ @ hX_Z + AP @ hX_P
              matmul(AP, hX_P, out=Q)
              Q[:] = dgemm(alpha=1.0, a=AZ, b=hX_Z, c=Q, overwrite_c=True, beta=1.0)
              Q[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=Q, overwrite_c=True, beta=1.0)
              copyto(AX, Q)
              # AP[:] = W
              copyto(AP, W)
            if B_products == 'implicit':
              #W[:] = BX @ hY_X + BZ @ hY_Z + BP @ hY_P
              matmul(BP, hY_P, out=W)
              W[:] = dgemm(alpha=1.0, a=BZ, b=hY_Z, c=W, overwrite_c=True, beta=1.0)
              W[:] = dgemm(alpha=1.0, a=BX, b=hY_X, c=W, overwrite_c=True, beta=1.0)
              #BX[:] = BX @ hX_X + BZ @ hX_Z + BP @ hX_P
              matmul(BP, hX_P, out=Q)
              Q[:] = dgemm(alpha=1.0, a=BZ, b=hX_Z, c=Q, overwrite_c=True, beta=1.0)
              Q[:] = dgemm(alpha=1.0, a=BX, b=hX_X, c=Q, overwrite_c=True, beta=1.0)
              copyto(BX, Q)
              # BP[:] = W
              copyto(BP, W)
            # W[:] = X @ hX_X + Z @ hX_Z + P @ hX_P
            matmul(P, hX_P, out=W)
            W[:] = dgemm(alpha=1.0, a=Z, b=hX_Z, c=W, overwrite_c=True, beta=1.0)
            W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
            # Q[:] = X @ hY_X + Z @ hY_Z + P @ hY_P
            matmul(P, hY_P, out=Q)
            Q[:] = dgemm(alpha=1.0, a=Z, b=hY_Z, c=Q, overwrite_c=True, beta=1.0)
            Q[:] = dgemm(alpha=1.0, a=X, b=hY_X, c=Q, overwrite_c=True, beta=1.0)
            # P[:] = Q
            copyto(P, Q)
            # X[:] = W
            copyto(X, W)
            if A_products != 'implicit':
              AX[:] = A @ X
              AP[:] = A @ P
            if B_products != 'implicit':
              BX[:] = B @ X
              BP[:] = B @ P
        if not skip_ortho:
          BZ[:] = B @ Z
          norm_BXP = sqrt(norm(BX)**2 + norm(BP)**2)
          OrthoB(Z, B, XP, BZ, norm_BXP)
          AZ[:] = A @ Z
          if modified:
            hX, Lambda, hY = RR5(X, Z, P, AX, AZ, AP, modified=True)
            hX_X, hX_Z, hX_P = hX[:m, :], hX[m:2*m, :], hX[2*m:3*m, :]
          else:
            hX, Lambda = RR5(X, Z, P, AX, AZ, AP)
            hX_X, hX_Z, hX_P = hX[:m, :], hX[m:2*m, :], hX[2*m:3*m, :]
            hY = vstack([zeros((m, m)), hX_Z, hX_P])
            Ortho(hY, vstack([hX_X, hX_Z, hX_P]))
          hY_X, hY_Z, hY_P = hY[:m, :], hY[m:2*m, :], hY[2*m:3*m, :]
          if A_products == 'implicit':
            # W[:] = AX @ hY_X + AZ @ hY_Z + AP @ hY_P
            matmul(AP, hY_P, out=W)
            W[:] = dgemm(alpha=1.0, a=AZ, b=hY_Z, c=W, overwrite_c=True, beta=1.0)
            W[:] = dgemm(alpha=1.0, a=AX, b=hY_X, c=W, overwrite_c=True, beta=1.0)
            # AX[:] = AX @ hX_X + AZ @ hX_Z + AP @ hX_P
            matmul(AP, hX_P, out=Q)
            Q[:] = dgemm(alpha=1.0, a=AZ, b=hX_Z, c=Q, overwrite_c=True, beta=1.0)
            Q[:] = dgemm(alpha=1.0, a=AX, b=hX_X, c=Q, overwrite_c=True, beta=1.0)
            copyto(AX, Q)
            # AP[:] = W
            copyto(AP, W)
          if B_products == 'implicit':
            #W[:] = BX @ hY_X + BZ @ hY_Z + BP @ hY_P
            matmul(BP, hY_P, out=W)
            W[:] = dgemm(alpha=1.0, a=BZ, b=hY_Z, c=W, overwrite_c=True, beta=1.0)
            W[:] = dgemm(alpha=1.0, a=BX, b=hY_X, c=W, overwrite_c=True, beta=1.0)
            #BX[:] = BX @ hX_X + BZ @ hX_Z + BP @ hX_P
            matmul(BP, hX_P, out=Q)
            Q[:] = dgemm(alpha=1.0, a=BZ, b=hX_Z, c=Q, overwrite_c=True, beta=1.0)
            Q[:] = dgemm(alpha=1.0, a=BX, b=hX_X, c=Q, overwrite_c=True, beta=1.0)
            copyto(BX, Q)
            # BP[:] = W
            copyto(BP, W)
          # W[:] = X @ hX_X + Z @ hX_Z + P @ hX_P
          matmul(P, hX_P, out=W)
          W[:] = dgemm(alpha=1.0, a=Z, b=hX_Z, c=W, overwrite_c=True, beta=1.0)
          W[:] = dgemm(alpha=1.0, a=X, b=hX_X, c=W, overwrite_c=True, beta=1.0)
          # Q[:] = X @ hY_X + Z @ hY_Z + P @ hY_P
          matmul(P, hY_P, out=Q)
          Q[:] = dgemm(alpha=1.0, a=Z, b=hY_Z, c=Q, overwrite_c=True, beta=1.0)
          Q[:] = dgemm(alpha=1.0, a=X, b=hY_X, c=Q, overwrite_c=True, beta=1.0)
          # P[:] = Q
          copyto(P, Q)
          # X[:] = W
          copyto(X, W)
          if A_products != 'implicit':
            AX[:] = A @ X
            AP[:] = A @ P
          if B_products != 'implicit':
            BX[:] = B @ X
            BP[:] = B @ P

      # R[:] = AX - BX @ diag(Lambda)
      copyto(R, AX)
      R[:] = dgemm(alpha=-1.0, a=BX, b=diag(Lambda), c=R, overwrite_c=True, beta=1.0)
      res[:, j] = norm(R, axis=0)
      res[:, j] /= abs(Lambda)
      print("extrema(res) =", res[:, j].min(), res[:, j].max())

      for i in range(k, nev):
        if res[i, j] < tol:
          k += 1
        else:
          break

      if k >= nev:
        return Lambda, X, res[:, :j+1]
            
  return Lambda, X, res[:, :j+1]