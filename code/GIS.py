import numpy as np

def GIS(X, assume_centered = False, verbose = False):
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
        deltahat_1 = (1-c)*invlambda+2*c*invlambda*theta
        delta = 1/((1-c)**2*invlambda+2*c*(1-c)*invlambda*theta+c**2*invlambda*Atheta2)
    else:
        if verbose:
            print("Necessary c <= 1 for Symmetrized KL divergence")        
        delta0 = 1/((c-1)*invlambda.mean())    #shrinkage of null eigenvalues
        delta = np.repeat(delta0,p-n)
        delta = np.concatenate((delta, 1/(invlambda*Atheta2)), axis=None)
        deltahat_1 = np.concatenate((delta[:p-n], (1-c)*invlambda+2*c*invlambda*theta), axis=None)

    x = np.min(invlambda)
    deltaLIS_1 = deltahat_1
    deltaLIS_1[deltaLIS_1 < x] = x
    
    deltaGIS = (delta/deltaLIS_1)**0.5
    
    sigmahat = (u @ np.diag(deltaGIS) @ u.T.conjugate()).real
    return sigmahat

def GIS_ewma(X, alpha, assume_centered = False, verbose = False):
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
        deltahat_1 = (1-c)*invlambda+2*c*invlambda*theta
         
        m_psi = theta + Htheta*1j
        m = -invlambda*m_psi
        c = p/n
        theta1 = (np.exp(alpha*c*(1+lambda1*m))-1)/beta/c/(1-np.exp(-alpha+alpha*c*(1+lambda1*m)))        
        s = (1+lambda1*m).imag/theta1.imag
        delta = lambda1/s
    else:
        if verbose:
            print("Necessary c <= 1 for Symmetrized KL divergence")        
        delta0 = 1/((c-1)*invlambda.mean())    #shrinkage of null eigenvalues
        delta = np.repeat(delta0,p-n)
        delta = np.concatenate((delta, 1/(invlambda*Atheta2)), axis=None)
        deltahat_1 = np.concatenate((delta[:p-n], (1-c)*invlambda+2*c*invlambda*theta), axis=None)

    x = np.min(invlambda)
    deltaLIS_1 = deltahat_1
    deltaLIS_1[deltaLIS_1 < x] = x
    
    deltaGIS = (delta/deltaLIS_1)**0.5
    
    sigmahat = (u @ np.diag(deltaGIS) @ u.T.conjugate()).real
    return sigmahat

def GIS_prec(X, assume_centered = False, verbose = False):
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
        delta = (1-c+2*c*theta)*invlambda
    else:
        raise ValueError("p <= n necessary for precision nl analytical shrinkage estimation.")

    x = np.min(invlambda)
    deltaLIS_1 = deltahat_1
    deltaLIS_1[deltaLIS_1 < x] = x
    
    deltaGIS = (delta*deltaLIS_1)**0.5
    
    psihat = (u @ np.diag(deltaGIS) @ u.T.conjugate()).real
    return psihat

def GIS_ewma_prec(X, alpha, assume_centered = False, verbose = False):
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
        deltahat_1 = (1-c)*invlambda+2*c*invlambda*theta
         
        m_psi = theta + Htheta*1j
        m = -invlambda*m_psi
        theta1 = (np.exp(alpha*c*(1+lambda1*m))-1)/beta/c/(1-np.exp(-alpha+alpha*c*(1+lambda1*m)))      
        s = (m*(1+lambda1*m)/theta1).imag/m.imag
        delta = s*invlambda
    else:
        raise ValueError("p <= n necessary for precision nl analytical shrinkage estimation.")

    x = np.min(invlambda)
    deltaLIS_1 = deltahat_1
    deltaLIS_1[deltaLIS_1 < x] = x
    
    deltaGIS = (delta*deltaLIS_1)**0.5
    
    psihat = (u @ np.diag(deltaGIS) @ u.T.conjugate()).real
    return psihat