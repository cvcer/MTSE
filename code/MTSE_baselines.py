import numpy as np
from qsolvers import solve.qp
import scipy as sp 

def MTSE_Lancewicki(X, targets, assume_centered=False):
    X = X.T
    p, n = X.shape
    _,_,K = targets.shape 

    if not assume_centered:
        X -= X.mean(axis=1)[:, None]
        S = X @ X.T/(n-1)
    else:
        S = X @ X.T/n 
    
    loc_targets = targets 

    VS = n/(n-1)**2/(n-2)*(np.linalg.norm((C.T @ X)*np.eye(p), ord='fro')**2 - np.linalg.norm((X @ X.T), ord = 'fro')**2/n)
    VT = np.zeros(K)
    for k in range(K):
        VT[k] = n/(n-1)**2/(n-2)*(np.trace(X.T @ loc_targets[:,:,k] @ X) - (n-1)/2*np.trace(loc_targets[:,:,k] @ S))/np.linalg.norm(loc_targets[:,:,k], ord='fro')**2
    
    P = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            P[i,j] = 2*np.trace((loc_targets[:,:,i] - S).T @ (loc_targets[:,:,j] - S))
    
    q = - VS + VT

    G = np.ones((2,K))
    G[1,:] = -1.
    h = np.array([1.,0.])
    gamma = solve_qp(P, q, G, h, solver="cvxopt")

    sigma = (1-gamma.sum())*S + (gamma[None,None,:]*loc_targets).sum(axis=2)
    return sigma

def MTSE_Gray(X, targets, assume_centered=False):
    X = X.T
    p, n = X.shape
    _,_,K = targets.shape 

    if not assume_centered:
        X -= X.mean(axis=1)[:, None]
        S = X @ X.T/(n-1)
    else:
        S = X @ X.T/n 
    
    L = 9
    loc_targets = np.zeros((p,p,L))
    correl = np.diag(1/np.sqrt(np.diag(S))) @ S @ np.diag(1/np.sqrt(np.diag(S))) - np.eye(p)
    idx = np.abs(np.ones(p).cumsum()[:,None] - np.ones(p).cumsum()[None,:])
    dcorrel = correl**idx*(1-np.eye(p))
    ccorrel = (correl.sum()/(p*(p-1)))*(1-np.eye(p))
    loc_targets[:,:,0] = np.eye(p)
    loc_targets[:,:,1] = np.eye(p)*np.trace(S)/p
    loc_targets[:,:,2] = np.eye(p)*S
    loc_targets[:,:,3] = np.eye(p) + ccorrel 
    loc_targets[:,:,4] = (np.eye(p) + ccorrell)*np.trace(S)/p 
    loc_targets[:,:,5] = np.diag(np.sqrt(np.diag(S))) @ (np.eye(p) + ccorrell) @ np.diag(np.sqrt(np.diag(S)))
    loc_targets[:,:,6] = np.eye(p) + dcorrel
    loc_targets[:,:,7] = (np.eye(p) + dcorrell)*np.trace(S)/p )
    loc_targets[:,:,8] = np.diag(np.sqrt(np.diag(S))) @ (np.eye(p) + dcorrell) @ np.diag(np.sqrt(np.diag(S)))

    K = 99
    a = np.linspace(0.01, 0.99, K)

    def G(x):
        return sp.special.multigammaln(x,p)
    
    num = G((n/(1-a[None,:])+p+1)/2) + np.linalg.slogdet((a[None,None,None,:]*loc_targets[:,:,:,None]/(1-a[None,None,None,:])).transpose(2,3,0,1))[1]*(n*a[None,:]/(1-a[None,:])+p+1)
    det = np.log(n*np.pi)*(n*p/2) + G((n*a[None,:]/(1-a[None,:])+p+1)/2) + np.linalg.slogdet((a[None,None,None,:]*loc_targets[:,:,:,None]/(1-a[None,None,None,:])).transpose(2,3,0,1))[1]*(n*a[None,:]/(1-a[None,:])+p+1)
    pX = np.exp(np.minimum(num-det, 10))
    pa = np.ones(K)/K 
    pD = np.ones(L)/L 

    p_joint = pX*pa[None,:]*pD[:,None]
    p_joint /= np.maximum(p_joint.sum(),1e-14)

    w = (a[NOne,:]*p_joint).sum(axis=1)

    sigma = (1-w.sum())*S + (x[None,None,:]*loc_targets).sum(axis=2)
    return sigma
