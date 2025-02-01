import numpy as np

def LIS(X, assume_centered = False, verbose = False):
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
        deltahat_1 = (1-c)*invlambda+2*c*invlambda*theta
    else:
        if verbose:
            print("Necessary c <= 1 for Stein's loss")
        delta0 = np.inf
        delta = np.repeat(delta0,p-n)
        deltahat_1 = np.concatenate((delta, (1-c)*invlambda+2*c*invlambda*theta), axis=None)
    
    x = np.min(invlambda)
    deltaLIS_1 = deltahat_1
    deltaLIS_1[deltaLIS_1 < x] = x
    deltaLIS = 1/deltaLIS_1
    
    sigmahat = (u @ np.diag(deltaLIS) @ u.T.conjugate()).real
    return sigmahat

def LIS_ewma(X, alpha, assume_centered = False, verbose = False):
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
        deltahat_1 = (1-c)*invlambda+2*c*invlambda*theta
    else:
        # TODO
        if verbose:
            print("Necessary c <= 1 for Stein's loss")
        delta0 = np.inf
        delta = np.repeat(delta0,p-n)
        deltahat_1 = np.concatenate((delta, (1-c)*invlambda+2*c*invlambda*theta), axis=None)
    
    x = np.min(invlambda)
    deltaLIS_1 = deltahat_1
    deltaLIS_1[deltaLIS_1 < x] = x
    deltaLIS = 1/deltaLIS_1
    
    sigmahat = (u @ np.diag(deltaLIS) @ u.T.conjugate()).real
    return sigmahat

def LIS_prec(X, assume_centered = False, verbose = False):
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
        deltahat_1 = (1-c)*invlambda+2*c*invlambda*theta
    else:
        raise ValueError("p <= n necessary for precision nl analytical shrinkage estimation.")
    
    x = np.min(invlambda)
    deltaLIS_1 = deltahat_1
    deltaLIS_1[deltaLIS_1 < x] = x
    
    psihat = (u @ np.diag(deltaLIS_1) @ u.T.conjugate()).real
    return psihat

def LIS_ewma_prec(X, alpha, assume_centered = False, verbose = False):
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
        deltahat_1 = (1-c)*invlambda+2*c*invlambda*theta
    else:
        raise ValueError("p <= n necessary for precision nl analytical shrinkage estimation.")
    
    x = np.min(invlambda)
    deltaLIS_1 = deltahat_1
    deltaLIS_1[deltaLIS_1 < x] = x
    
    psihat = (u @ np.diag(deltaLIS_1) @ u.T.conjugate()).real
    return psihat