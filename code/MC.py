import numpy as np
import pickle
import os
import datetime as dt
from scipy.stats import multivariate_t
import warnings

from dataio import *

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance
from LWO_estimator import *
from MTSE import *

def targets_gen_id():
    def targets_gen(n, p):
        return np.eye(p)[:,:,np.newaxis]
    return targets_gen

def targets_gen_2blocks():
    def targets_gen(n, p):
        targets = np.zeros((p,p,2))
        targets[:,:,0] = np.eye(p)
        targets[:p//2,:p//2,1] = np.eye(p//2)
        return targets
    return targets_gen

def targets_gen_2unblocks():
    def targets_gen(n, p):
        targets = np.zeros((p,p,2))
        targets[:,:,0] = np.eye(p)
        targets[p//4:3*(p//4),p//4:3*(p//4),1] = np.eye(2*(p//4))
        return targets
    return targets_gen

def cov_gen_wishart(perturb_coeff = 0.0, cov_cst = None):
    def cov_gen(n, p):
        try:
            if cov_cst.shape[0] < p or cov_cst.shape[1] < p:
                cov_c = np.eye(p)
            else:
                cov_c = cov_cst[:p,:p]
        except:
            cov_c = np.eye(p)
        
        X_perturb = np.random.multivariate_normal(np.zeros(p), cov_c, size =p).T
        cov_perturb = X_perturb @ X_perturb.T/p
        cov_perturb /= np.sqrt(np.sqrt(np.linalg.norm(cov_perturb @ cov_perturb.T, ord='fro')**2/p))
        cov = (1 - perturb_coeff) * cov_c + perturb_coeff * cov_perturb
        return cov
    return cov_gen

def cov_gen_2blocks(coeff = 5.0):
    def cov_gen(n, p):
        cov_cst = np.eye(p)        
        X_perturb1 = np.random.multivariate_normal(np.zeros(p//2), cov_cst[:p//2, :p//2], size = p//2).T
        cov_perturb1 = X_perturb1 @ X_perturb1.T/(p//2)
        cov_perturb1 /= np.sqrt(np.sqrt(np.linalg.norm(cov_perturb1 @ cov_perturb1.T, ord='fro')**2/p))
        
        X_perturb2 = np.random.multivariate_normal(np.zeros(p-p//2), cov_cst[p//2:, p//2:], size = p-p//2).T
        cov_perturb2 = X_perturb2 @ X_perturb2.T/(p-p//2)
        cov_perturb2 /= np.sqrt(np.sqrt(np.linalg.norm(cov_perturb2 @ cov_perturb2.T, ord='fro')**2/p))
        
        cov = np.zeros((p,p))
        cov[:p//2,:p//2] = coeff*cov_perturb1
        cov[p//2:,p//2:] = cov_perturb2
        return cov
    return cov_gen

def cov_gen_Kblocks(K, coeff):
    def cov_gen(n, p):
        cov = np.zeros((p,p))
        cov_cst = np.eye(p)        
        for i in range(K):
            X_perturb = np.random.multivariate_normal(np.zeros(p//K), cov_cst[:p//K, :p//K], size = p//K).T
            cov_perturb = X_perturb @ X_perturb.T/(p//K)
            cov[i*(p//K):(i+1)*(p//K),i*(p//K):(i+1)*(p//K)] = cov_perturb*coeff[i]
        idx = K*(p//K)
        if p - idx != 0:
            X_perturb = np.random.multivariate_normal(np.zeros(p-idx), cov_cst[:p-idx, :p-idx], size = p-idx).T
            cov_perturb = X_perturb @ X_perturb.T/(p-idx)
            cov[idx:,idx:] = cov_perturb
        
        cov /= np.sqrt(np.sqrt(np.linalg.norm(cov @ cov.T, ord='fro')**2/p))
        return cov
    return cov_gen

def targets_gen_kKblocks(k, K, offset = 0):
    def targets_gen(n, p):
        if offset == 1:
            targets = np.zeros((p,p,k))
            targets[:,:,0] = np.eye(p)
            for i in range(1,min(k,K)):
                idx = i*(p//K)
                targets[idx:idx+(p//(2*K)),idx:idx+(p//(2*K)),i] = np.eye(p//(2*K))
            for i in range(1,min(k,K)):
                idx = (K-i)*(p//K)
                targets[idx+(p//(2*K)):idx+(p//K),idx+(p//(2*K)):idx+(p//K),i] = np.eye(p//K - p//(2*K))
            for i in range(K,k):
                X_perturb = np.random.multivariate_normal(np.zeros(p), np.eye(p), size =p).T
                cov_perturb = X_perturb @ X_perturb.T/p
                targets[:,:,i] = cov_perturb
        else:
            targets = np.zeros((p,p,k))
            targets[:,:,0] = np.eye(p)
            for i in range(min(k,K)-1):
                idx = i*(p//K)
                targets[idx:idx+(p//K),idx:idx+(p//K),i+1] = np.eye(p//K)
            for i in range(K-1, k-1):
                X_perturb = np.random.multivariate_normal(np.zeros(p), np.eye(p), size =p).T
                cov_perturb = X_perturb @ X_perturb.T/p
                targets[:,:,i+1] = cov_perturb
        return targets
    return targets_gen

def targets_gen_rd(K):
    def targets_gen(n, p):
        targets = np.zeros((p,p,K))
        targets[:,:,0] = np.eye(p)
        for i in range(1,K):
            X_perturb = np.random.multivariate_normal(np.zeros(p), np.eye(p), size =p).T
            cov_perturb = X_perturb @ X_perturb.T/p
            targets[:,:,i] = cov_perturb
        return targets
    return targets_gen

def MC(n_mc, p, n, cov_gen, targets_gen, nu = np.inf, assume_centered = False):
    loss = np.zeros((12, n_mc))
    
    targets = targets_gen(n, p)
    targets, free_indices, P = gram_schmidt(targets)
    S_r = targets[:,:,0]
    for i in range(n_mc):
        cov = cov_gen(n, p)
        
        if nu == np.inf:
            X = np.random.multivariate_normal(np.zeros(p), cov, size = n)
        else:
            sigma = (nu-2)/nu*cov
            rv = multivariate_t(loc = np.zeros(p), shape = sigma, df = nu, allow_singular = True)
            X = rv.rvs(size = n)
        
        S_EC = EmpiricalCovariance(assume_centered = assume_centered, store_precision = False).fit(X).covariance_
        S_LW = LedoitWolf(assume_centered = assume_centered, store_precision = False).fit(X).covariance_
        S_OAS = OAS(assume_centered = assume_centered, store_precision = False).fit(X).covariance_
        
        S_LWO = LWO_estimator(X, S_r = None, assume_centered = assume_centered)
        S_LWO_oracle = LWO_estimator_oracle(X, cov, S_r = None, assume_centered = assume_centered)
        S_MTSE = MTSE_estimator(X, targets, assume_orthonormal = True, assume_centered = assume_centered)
        S_MTSE_oracle = MTSE_estimator_oracle(X, targets, cov, assume_centered = assume_centered)
        
        loss[0,i] = np.linalg.norm(S_EC - cov, ord='fro')**2/p
        loss[2,i] = np.linalg.norm(S_LW - cov, ord='fro')**2/p
        loss[3,i] = np.linalg.norm(S_OAS - cov, ord='fro')**2/p
        loss[6,i] = np.linalg.norm(S_LWO - cov, ord='fro')**2/p
        loss[7,i] = np.linalg.norm(S_LWO_oracle - cov, ord='fro')**2/p
        loss[8,i] = np.linalg.norm(S_MTSE - cov, ord='fro')**2/p
        loss[9,i] = np.linalg.norm(S_MTSE_oracle - cov, ord='fro')**2/p
        loss[11,i] = np.linalg.norm(cov, ord='fro')**2/p
    return loss

def MCs(n_mc, pn_list, cov_gen, targets_gen, nu = np.inf, assume_centered = False, PATH = "", verbose = False):
    for (p,n) in pn_list:
        try:
            loss = MC(n_mc, p, n, cov_gen, targets_gen, nu = nu, assume_centered = False)
            if verbose:
                print(loss[:,0])
            export_to_pkl(n_mc, p, n, cov_gen, targets_gen, nu, loss, PATH, verbose = verbose)
        except:
            print("[", p, n, "] failed.")























