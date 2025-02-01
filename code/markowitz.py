import numpy as np

def get_S(X):
    n,p = X.shape
    S = np.cov(X.T)
    return S

def get_IC(e):
    if e.std() != 0:
        IC = np.sqrt(252)*e.mean()/e.std()
    else:
        IC = 0
    return IC

def get_e(X, x):
    e = X @ x
    return e

def GMV(X_test, Sigma, wb):
    n,p = X_test.shape
    one = np.ones(p)
    try:
        x = np.linalg.pinv(Sigma) @ one/(np.ones((1,p)) @ np.linalg.pinv(Sigma) @ one)
    except:
        x = one/p
    e = get_e(X_test, x)
    s = e.var()
    IC = get_IC(e)
    return e.sum(), s, IC