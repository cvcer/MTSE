import numpy as np

def QIS(X, assume_centered = False):
    n,p = X.shape

    #default setting
    if assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]        
        n = n-1

    c = p/n 
    sample = X.T @ X/n  
    sample = (sample+sample.T)/2                              #make symmetrical

    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)    #use Cholesky factorisation based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)    #reset negative values to 0

    #COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2,1/c**2)**0.35)/p**0.35    #smoothing parameter
    invlambda = 1/lambda1[max(1,p-n+1)-1:p]    #inverse of (non-null) eigenvalues
    
    Lj = np.repeat(invlambda[:,np.newaxis], min(p,n), axis=1)
    Lj_i = Lj - Lj.T
    
    theta = (Lj*Lj_i/(Lj_i**2 + Lj**2*h**2)).mean(axis=0)    #smoothed Stein shrinker 
    Htheta = (Lj**2*h/(Lj_i**2 + Lj**2*h**2)).mean(axis=0)    #its conjugate
    Atheta2 = theta**2+Htheta**2    #its squared amplitude
    
    if p<=n:    #case where sample covariance matrix is not singular
        delta = 1 / ((1-c)**2*invlambda+2*c*(1-c)*invlambda*theta +c**2*invlambda*Atheta2)    #optimally shrunk eigenvalues
    else:
        delta0 = 1/((c-1)*invlambda.mean())    #shrinkage of null eigenvalues
        delta = np.repeat(delta0,p-n)
        delta = np.concatenate((delta, 1/(invlambda*Atheta2)), axis=None)

    deltaQIS = delta*(lambda1.sum()/delta.sum())    #preserve trace
    
    sigmahat = (u @ np.diag(deltaQIS) @ u.T.conjugate()).real
    return sigmahat
        
def QIS_ewma(X, alpha, assume_centered = False):
    n,p = X.shape

    #default setting
    if assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]        
        n = n-1

    c = p/n 
    beta = alpha/(1-np.exp(-alpha))
    sample = X.T @ X/n  
    sample = (sample+sample.T)/2                              #make symmetrical

    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)    #use Cholesky factorisation based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)    #reset negative values to 0

    #COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2,1/c**2)**0.35)/p**0.35    #smoothing parameter
    invlambda = 1/lambda1[max(1,p-n+1)-1:p]    #inverse of (non-null) eigenvalues
    
    Lj = np.repeat(invlambda[:,np.newaxis], min(p,n), axis=1)
    Lj_i = Lj - Lj.T
    
    theta = (Lj*Lj_i/(Lj_i**2 + Lj**2*h**2)).mean(axis=0)    #smoothed Stein shrinker 
    Htheta = (Lj**2*h/(Lj_i**2 + Lj**2*h**2)).mean(axis=0)    #its conjugate
    Atheta2 = theta**2+Htheta**2    #its squared amplitude
    
    if p<=n:    #case where sample covariance matrix is not singular        
        m_psi = theta + Htheta*1j
        m = -invlambda*m_psi
        c = p/n
        theta1 = (np.exp(alpha*c*(1+lambda1*m))-1)/beta/c/(1-np.exp(-alpha+alpha*c*(1+lambda1*m)))        
        s = (1+lambda1*m).imag/theta1.imag
        delta = lambda1/s
    else:
        delta0 = 1/((c-1)*invlambda.mean())    #shrinkage of null eigenvalues
        delta = np.repeat(delta0,p-n)
        delta = np.concatenate((delta, 1/(invlambda*Atheta2)), axis=None)

    deltaQIS = delta*(lambda1.sum()/delta.sum())    #preserve trace
    
    sigmahat = (u @ np.diag(deltaQIS) @ u.T.conjugate()).real
    return sigmahat

def QIS_prec(X, assume_centered = False):
    n,p = X.shape

    #default setting
    if assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]        
        n = n-1

    c = p/n 
    sample = X.T @ X/n  
    sample = (sample+sample.T)/2                              #make symmetrical

    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)    #use Cholesky factorisation based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)    #reset negative values to 0

    #COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2,1/c**2)**0.35)/p**0.35    #smoothing parameter
    invlambda = 1/lambda1[max(1,p-n+1)-1:p]    #inverse of (non-null) eigenvalues
    
    Lj = np.repeat(invlambda[:,np.newaxis], min(p,n), axis=1)
    Lj_i = Lj - Lj.T
    
    theta = (Lj*Lj_i/(Lj_i**2 + Lj**2*h**2)).mean(axis=0)    #smoothed Stein shrinker 
    
    if p<=n:    #case where sample covariance matrix is not singular
        delta = (1-c+2*c*theta)*invlambda
    else:
        raise ValueError("p <= n necessary for precision nl analytical shrinkage estimation.")

    deltaQIS = delta
    
    psihat = (u @ np.diag(delta) @ u.T.conjugate()).real
    return psihat
        
def QIS_ewma_prec(X, alpha, assume_centered = False):
    n,p = X.shape

    #default setting
    if assume_centered:
        X -= X.mean(axis=0)[np.newaxis,:]        
        n = n-1

    c = p/n 
    beta = alpha/(1-np.exp(-alpha))
    sample = X.T @ X/n  
    sample = (sample+sample.T)/2                              #make symmetrical

    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)    #use Cholesky factorisation based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)    #reset negative values to 0

    #COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2,1/c**2)**0.35)/p**0.35    #smoothing parameter
    invlambda = 1/lambda1[max(1,p-n+1)-1:p]    #inverse of (non-null) eigenvalues
    
    Lj = np.repeat(invlambda[:,np.newaxis], min(p,n), axis=1)
    Lj_i = Lj - Lj.T
    
    theta = (Lj*Lj_i/(Lj_i**2 + Lj**2*h**2)).mean(axis=0)    #smoothed Stein shrinker 
    Htheta = (Lj**2*h/(Lj_i**2 + Lj**2*h**2)).mean(axis=0)    #its conjugate
    
    if p<=n:    #case where sample covariance matrix is not singular
        m_psi = theta + Htheta*1j
        m = -invlambda*m_psi
        theta1 = (np.exp(alpha*c*(1+lambda1*m))-1)/beta/c/(1-np.exp(-alpha+alpha*c*(1+lambda1*m)))      
        s = (m*(1+lambda1*m)/theta1).imag/m.imag
        delta = s*invlambda
    else:
        raise ValueError("p <= n necessary for precision nl analytical shrinkage estimation.")

    
    psihat = (u @ np.diag(delta) @ u.T.conjugate()).real
    return psihat