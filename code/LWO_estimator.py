import numpy as np

def LWO_estimator_oracle(X, cov, assume_centered = False, S_r = None, est = False):
    X = X.T
    p, n = X.shape
    Id = np.eye(p)
    
    if not assume_centered:
        X -= X.mean(axis=1)[:, np.newaxis]
        S = X @ X.T/(n-1)
    else:
        S = X @ X.T/n
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum() == 0:
            S_r = Id
    S_r /= np.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    v = np.concatenate([S[:,:,np.newaxis], S_r[:,:,np.newaxis]], axis=2)
    A = (v[:,:,:,np.newaxis] * v[:,:,np.newaxis,:]).sum(axis=1).sum(axis=0)/p
    b = (v * cov[:,:,np.newaxis]).sum(axis=1).sum(axis=0)/p
    c = np.linalg.solve(A, b)
    S_star = c[0]*S + c[1]*S_r
      
    if est:
        return S_star, c
    return S_star


def LWO_estimator(X, assume_centered = False, S_r = None, est = False):
    X = X.T
    if est:
      if not assume_centered:
          S_star, c = LWO_estimator_unknown_mean(X, S_r, est)
      else:
          S_star, c = LWO_estimator_known_mean(X, S_r, est)
      return S_star, c
    if not assume_centered:
        S_star = LWO_estimator_unknown_mean(X, S_r, est)
    else:
        S_star = LWO_estimator_known_mean(X, S_r, est)
    return S_star

def LWO_estimator_known_mean(X, S_r = None, est = False):
    p, n = X.shape
    Id = np.eye(p)
    S = X @ X.T/n
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum() == 0:
            S_r = Id
    S_r /= np.linalg.norm(S_r, ord = 'fro')**2/np.sqrt(p)
    
    mu = np.trace(S @ S_r.T)/p
    delta2 = np.linalg.norm(S-mu*S_r, ord = 'fro')**2/p
    beta2 = (((X**2).sum(axis=0)**2).sum()/n - np.linalg.norm(S, ord = 'fro')**2)/p/n
    beta2 = n/(n-1)*beta2
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    if est:
        c = np.array([1-shrinkage, shrinkage*mu])
        return shrinkage*mu*S_r + (1 - shrinkage)*S, c
    return shrinkage*mu*S_r + (1 - shrinkage)*S

def LWO_estimator_unknown_mean(X, S_r = None, est = False):
    p, n = X.shape
    Id = np.eye(p)
    X -= X.mean(axis=1)[:,np.newaxis]
    S = X @ X.T/(n-1)
    
    try:
        if S_r == None:
            S_r = Id
    except ValueError:
        if (S_r**2).sum() == 0:
            S_r = Id
    S_r /= np.linalg.norm(S_r, ord = 'fro')/np.sqrt(p)
    
    mu = np.trace(S @ S_r.T)/p
    delta2 = np.linalg.norm(S-mu*S_r, ord = 'fro')**2/p
    
    mu_I = np.trace(S)/p
    S2 = np.linalg.norm(S, ord = 'fro')**2/p
    beta2_bar = ((X**2).sum(axis=0)**2).sum()/p/(n-1)**2 - np.linalg.norm(S, ord = 'fro')**2/p/n
    beta2 = (n-1)**2/(n-2)/(n-3)*beta2_bar- 1/n/(n-2)*S2 - (n-1)/n/(n-2)/(n-3)*p*mu_I**2
    beta2 = max(beta2, 0)
    beta2 = min(beta2, delta2)
    
    shrinkage = beta2/delta2
    if est:
        c = np.array([1-shrinkage, shrinkage*mu])
        return shrinkage*mu*S_r + (1 - shrinkage)*S, c
    return shrinkage*mu*S_r + (1 - shrinkage)*S