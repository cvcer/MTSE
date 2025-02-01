import numpy as np
import pickle
import os
import datetime as dt
from scipy.stats import multivariate_t
import warnings

def export_to_pkl(n_mc, p, n, cov_gen, targets_gen, nu, loss, PATH, verbose = False):
    filename = "MC_"+str(n_mc)+"_"+str(p)+"_"+str(n)+"_"+dt.datetime.today().strftime(format = "%Y%m%d_%H%M%S")+".pkl"
    try:
        os.mkdir(PATH)
    except OSError as error:
        if verbose:
            print(error)
    path = os.path.join(PATH, filename)
    
    dic = {
        'params':[n_mc, p, n, nu],
        'loss':loss
        }
    with open(path, 'wb') as file:
        pickle.dump(dic, file)

def MC_loader(verbose = True, PATH = ""):
    try:
        files = os.listdir(PATH)
        files.sort()
    except:
        files = [""]
        
    params_array = np.zeros((4,0))
    loss_mean = np.zeros((12,0))
    loss_std = np.zeros((12,0))
    
    for file_name in files:
        if file_name != "":
            path_file = os.path.join(PATH, file_name)
        else:
            path_file = PATH
        with open(path_file, 'rb') as file:
            dic = pickle.load(file)
            params = dic['params']
            loss = dic['loss']
            
            p, n = params[1], params[2]
            params_array = np.concatenate([params_array, np.array(params)[:,np.newaxis]], axis=1)
            
            loss_mean = np.concatenate([loss_mean, loss.mean(axis=1)[:,np.newaxis]], axis=1)
            loss_std = np.concatenate([loss_std, ((loss - loss[9,None,:])/np.sqrt(loss.shape[1])).std(axis=1)[:,np.newaxis]], axis=1)
    return params_array, loss_mean, loss_std