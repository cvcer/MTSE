import numpy as np

def AS_estimator(X, assume_centered = False):
    n, p = X.shape
    if not assume_centered:
        X -= X.mean(axis=0)[np.newaxis, :]
        n -= 1
    sample = X.T @ X/n
    lambda_, u = np.linalg.eigh(sample)
    lambda_ = lambda_[max(0, p-n):p]
    L = np.repeat(lambda_[:,np.newaxis], min(p,n), axis=1)
    h = n**(-1/3)
    H = h*L.T
    x = (L - L.T)/H
    ftilde = (3/4/np.sqrt(5))*np.mean(np.maximum(1-x**2/5,0)/H, axis=1)
    Hftemp = (-3/10/np.pi)*x + (3/4/np.sqrt(5)/np.pi)*(1-x**2/5)*np.log(np.abs((np.sqrt(5)-x)/(np.sqrt(5)+x)))
    Hftemp[np.abs(x) == np.sqrt(5)] = (-3/10/np.pi)*x[np.abs(x) == np.sqrt(5)]
    Hftilde = (Hftemp/H).mean(axis=1)
    if p <= n:
        dtilde = lambda_/((np.pi*(p/n)*lambda_*ftilde)**2 + (1-p/n-np.pi*p/n*lambda_*Hftilde)**2)
    else:
        Hftilde0 = (1/np.pi)*(3/10/h**2 + 3/4/np.sqrt(5)/h*(1-1/5/h**2)*np.log((1+np.sqrt(5)*h)))*(1/lambda_).mean()
        dtilde0 = 1/(np.pi*(p-n)/n*Hftilde0)
        dtilde1 = lambda_/(np.pi**2*lambda_**2*(ftilde**2+Hftilde**2))
        dtilde = np.concatenate([dtilde0*np.ones(p-n), dtilde1], axis=0)
    sigmatilde = u @ np.diag(dtilde) @ u.T
    return sigmatilde