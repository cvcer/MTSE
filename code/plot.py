import numpy as np
import pandas as pd
import pickle
import os
import datetime as dt
from scipy.stats import multivariate_t
import warnings

from dataio import *
from MC import *
from plot import *

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance
from LWO_estimator import *
from MTSE import *
from analytical_shrinkage import *

import matplotlib.pyplot as plt

def plot_usefulness(PATH=".\\rst\\exp_usefulness\\"):
    K = 10
    plt.figure()
    for offset in [0,1]:
        Y1 = []
        Y2 = []
        Y3 = []
        for k in range(1,11):
            params_array, loss_mean, loss_std = MC_loader(verbose = False, PATH = PATH+"exp_usefulness_"+str(K)+"_"+str(k)+"_"+str(offset)+"\\")
            Y1 += [(loss_mean[0] - loss_mean[8])/loss_mean[0]]
            Y2 += [(loss_mean[0] - loss_mean[9])/loss_mean[0]]
            Y3 += [(loss_mean[0] - loss_mean[6])/loss_mean[0]]
        Y1 = np.array(Y1)
        Y2 = np.array(Y2)
        Y3 = np.array(Y3)
        
        xlabel = "Number of targets"
        X = np.ones(10).cumsum() 
        
        if offset == 0:
            plt.plot(X, Y1[:,0], label="MTSE_1, aligned")
            
            plt.plot(X, Y2[:,0], label="MTSE_1 oracle,aligned")
        else:
            plt.plot(X, Y1[:,0], label="MTSE_2, misaligned")
            
            plt.plot(X, Y2[:,0], label="MTSE_2 oracle, misaligned")
            
    plt.plot(X, np.repeat(Y3.mean(), 10), label="LW")
        
    plt.xlabel(xlabel)
    plt.ylabel("PRIAL")
    plt.title("")
    plt.legend()
    plt.show()
        
def plot_rdtargets(PATH=".\\rst\\exp_rdtargets\\"):
    xlabel = "Number of targets"
    X = np.ones(20).cumsum()
    Y = np.zeros([12,20])
    
    plt.figure()
    for k in range(1,21):
        foldername = "exp_rdtargets_"+str(k)
        params_array, loss_mean, loss_std = MC_loader(verbose = False, PATH = PATH+foldername)
        Y[:,k-1] = (loss_mean[6,0,None] - loss_mean[:,0])/loss_mean[0,0,None]
        
    plt.plot(X, Y[8], label="MTSE - LW")
    
    plt.plot(X, Y[9], label="MTSE_o - LW")
        
    plt.xlabel(xlabel)
    plt.ylabel("PRIAL")
    plt.title("")
    plt.legend()
    
    plt.show()

def plot_heavy(PATH=".\\rst\\exp_heavys\\"):
    plt.figure()
    plt.yscale('log',base=10) 
    for nu in [20000, 8, 4, 3, 2.5]:
        foldername = "exp_heavy_5_5_0_"+str(nu)
        params_array, loss_mean, loss_std = MC_loader(verbose = False, PATH = PATH + foldername)
        
        xlabel = "Number of dimensions"
        X = np.ones(params_array.shape[1]).cumsum()*2 - 1
        
        
        Y = loss_mean[8] - loss_mean[9]
        e = loss_std[8] # we stored in it the std of loss[8,:] - loss[9,:], as wanted
        plt.errorbar(X, Y, e, label="nu="+str(nu))
    
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Frobenius norm")
    plt.title("")
    plt.legend()
    plt.show()
    

























