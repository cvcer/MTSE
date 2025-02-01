import numpy as np
import scipy as sp

def MTSE_estimator_oracle(X, targets, cov, assume_centered = False, assume_orthogonal = False):
    """
    

    Parameters
    ----------
    X : Numpy array of shape (n,p)
        The data matrix of n observations of dimension p.
    targets : Numpy array of shape (p,p,K)
        Array of K target matrices of shape (p,p), the set of targets.
    cov : Numpy array of shape (p,p)
        Population covariance.
    assume_centered : boolean, optional
        If data have been empirically demeaned or need to be demeaned, set False. 
        Set True only if the exact mean is known and has been removed.
        The default is False.
    assume_orthogonal : boolean, optional
        Set True if the set of targets has been orthogonalized. 
        Otherwise, it will be orthogonalized through Gram-Shcmidt algorithm. 
        The default is False.

    Returns
    -------
    Numpy array of shape (p,p)
        The oracle Multi-Target Linear Shrinkage estimator.

    """
    X = X.T
    p, n = X.shape
    _,_,K = targets.shape
    
    if not assume_centered:
        X -= X.mean(axis=1)[:, np.newaxis]
        S = X @ X.T/(n-1)
    else:
        S = X @ X.T/n
    
    v = np.concatenate([S[:,:,np.newaxis], targets], axis=2)
    A = (v[:,:,:,np.newaxis] * v[:,:,np.newaxis,:]).sum(axis=1).sum(axis=0)/p
    b = (v * cov[:,:,np.newaxis]).sum(axis=1).sum(axis=0)/p
    c = np.linalg.solve(A, b)
    S_star = c[0]*S + (c[1:][np.newaxis, np.newaxis, :]*targets).sum(axis=2)
    
    w, v = np.linalg.eigh(S_star)
    wp = w * (w >= 0)
    S_starp = v.real @ np.diag(wp.real) @ v.real.T
    return S_starp

def gram_schmidt(targets):
    p,_,K = targets.shape
    free_indices = np.zeros(0, dtype=int)
    tilde_targets = np.zeros((p,p,0))
    for i in range(K):
        temp_target = targets[:,:,i]
        for j in range(tilde_targets.shape[2]):
            temp_target -= (temp_target*tilde_targets[:,:,j]).sum()/p*tilde_targets[:,:,j]
        temp_target_norm = np.linalg.norm(temp_target, ord='fro')/np.sqrt(p)
        if temp_target_norm != 0:
            temp_target /= temp_target_norm
            tilde_targets = np.concatenate([tilde_targets, temp_target[:,:,np.newaxis]], axis=2)
            free_indices = np.append(free_indices, i)
    P = (tilde_targets[:,:,:,np.newaxis] * targets[:,:,np.newaxis,free_indices]).sum(axis=1).sum(axis=0)/p
    return tilde_targets, free_indices, P

def MTSE_estimator(X, targets, assume_centered = False, assume_orthonormal = False):
    """
    

    Parameters
    ----------
    X : Numpy array of shape (n,p)
        The data matrix of n observations of dimension p.
    targets : Numpy array of shape (p,p,K)
        Array of K target matrices of shape (p,p), the set of targets.
    cov : Numpy array of shape (p,p)
        Population covariance.
    assume_centered : boolean, optional
        If data have been empirically demeaned or need to be demeaned, set False. 
        Set True only if the exact mean is known and has been removed.
        The default is False.
    assume_orthogonal : boolean, optional
        Set True if the set of targets has been orthogonalized. 
        Otherwise, it will be orthogonalized through Gram-Shcmidt algorithm. 
        The default is False.

    Returns
    -------
    Numpy array of shape (p,p)
        The Multi-Target Linear Shrinkage estimator.
    
    """

    X = X.T
    if not assume_centered:
        S_star = MTSE_estimator_unknown_mean(X, targets, assume_orthonormal)
    else:
        S_star = MTSE_estimator_known_mean(X, targets, assume_orthonormal)
    return S_star

def MTSE_estimator_known_mean(X, targets, assume_orthonormal):
    if not assume_orthonormal:
        tilde_targets, free_indices, P = gram_schmidt(targets)
    else:
        tilde_targets = targets
        free_indices = np.ones(targets.shape[2], dtype=int).cumsum() -1
        P = np.eye(free_indices.shape[0])
    
    p, n = X.shape
    _,_,K = tilde_targets.shape
    S = X @ X.T/n
    
    S2 = np.linalg.norm(S, ord='fro')**2/p
    ST = (S[:,:,np.newaxis]*tilde_targets).sum(axis=1).sum(axis=0)/p
    
    vs = 1/(n*(n-1)*p)*(((X**2).sum(axis=0)**2).sum() - n*np.linalg.norm(S, ord='fro')**2)
    vt = np.zeros(K)
    for k in range(K):
        Tk = tilde_targets[:,:,k]
        sqrtTk = sp.linalg.sqrtm(Tk, disp=False)[0]
        Z = sqrtTk @ X
        vt[k] = (((Z**2).sum(axis=0)**2).sum() - n*(S*Tk).sum(axis=1).sum(axis=0)**2).real/(n*(n-1)*p**2)
    
    detA = S2 - ((S[:,:,np.newaxis]*tilde_targets).sum(axis=1).sum(axis=0)**2).sum()/p**2
    
    c1 = 1/detA*(S2 - vs - (ST**2 - vt).sum())
    c1 = np.minimum(np.maximum(0, c1), 1)
    c2 = (1 - c1)*ST
    
    S_star = c1*S + (c2[np.newaxis, np.newaxis, :]*tilde_targets).sum(axis=2)
    
    w, v = np.linalg.eigh(S_star)
    wp = w * (w >= 0)
    S_starp = v.real @ np.diag(wp.real) @ v.real.T
    
    return S_starp

def MTSE_estimator_unknown_mean(X, targets, assume_orthonormal):
    if not assume_orthonormal:
        tilde_targets, free_indices, P = gram_schmidt(targets)
    else:
        tilde_targets = targets
        free_indices = np.ones(targets.shape[2], dtype=int).cumsum() -1
        P = np.eye(free_indices.shape[0])
    
    p, n = X.shape
    _,_,K = tilde_targets.shape
    X -= X.mean(axis=1)[:,np.newaxis]
    S = X @ X.T/(n-1)
    
    mu_I = np.trace(S)/p
    S2 = np.linalg.norm(S, ord='fro')**2/p
    ST = (S[:,:,np.newaxis]*tilde_targets).sum(axis=1).sum(axis=0)/p
    
    beta2_bar = ((X**2).sum(axis=0)**2).sum()/p/(n-1)**2 - np.linalg.norm(S, ord = 'fro')**2/p/n
    vs = (n-1)**2/(n-2)/(n-3)*beta2_bar- 1/n/(n-2)*S2 - (n-1)/n/(n-2)/(n-3)*p*mu_I**2
    
    vt = np.zeros(K)
    for k in range(K):
        Tk = tilde_targets[:,:,k]
        sqrtTk = sp.linalg.sqrtm(Tk, disp=False)[0]
        Z = sqrtTk @ X
        beta2_barT = ((Z**2).sum(axis=0)**2).sum().real/p/(n-1)**2 - np.trace(S @ Tk @ S @ Tk)/p/n
        STk2s = ((S*Tk).sum(axis=1).sum(axis=0)/p)**2
        STk2n = np.trace(S @ Tk @ S @ Tk)/p
        
        vtk = beta2_barT*(n+2/(n-3)) + STk2n*(1-2/n) - STk2s*p*(n-2)*(n**2-2*n-1)/n/(n-1)/(n-3)
        vtk *= (n-1)/p/(n-2)**2
        vt[k] = vtk
    
    detA = S2 - ((S[:,:,np.newaxis]*tilde_targets).sum(axis=1).sum(axis=0)**2).sum()/p**2
    
    c1 = 1/detA*(S2 - vs - (ST**2 - vt).sum())
    c1 = np.minimum(np.maximum(0, c1), 1)
    c2 = (1 - c1)*ST
    
    S_star = c1*S + (c2[np.newaxis, np.newaxis, :]*tilde_targets).sum(axis=2)
    
    w, v = np.linalg.eigh(S_star)
    wp = w * (w >= 0)
    S_starp = v.real @ np.diag(wp.real) @ v.real.T
    
    return S_starp
































