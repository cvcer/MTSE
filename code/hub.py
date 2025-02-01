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

from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS
from LWO_estimator import *
from MTSE import *
from MTSE_baselines import MTSE_Lancewicki, MTSE_Gray
import GIS, QIS, LIS, analytical_shrinkage


from markowitz import *
from dataloader import *
import matplotlib.pyplot as plt

import time

def exp_usefulness(n_mc, PATH):
    pn_list = [(50,25)]
    nu = 9
    K = 10
    coeff = (K - np.ones(K).cumsum() + 1)*2
    cov_gen = cov_gen_Kblocks(K, coeff)
    
    verbose = False
    try:
        os.mkdir(PATH)
    except OSError as error:
        if verbose:
            print(error)
    
    for offset in [0,1]: # offset == 0 for aligned targets, offset == 1 for misaligned
        if offset == 0:
            for k in range(1,K+1):
                targets_gen = targets_gen_kKblocks(k, K, offset)
                foldername = "exp_usefulness_"+str(K)+"_"+str(k)+"_"+str(offset)
                MCs(n_mc, pn_list, cov_gen, targets_gen, nu = nu, assume_centered = False, PATH = PATH+foldername, verbose = False) 
        else:
            for k in range(1,K+1):
                targets_gen = targets_gen_kKblocks(k, K, offset)
                foldername = "exp_usefulness_"+str(K)+"_"+str(k)+"_"+str(offset)
                MCs(n_mc, pn_list, cov_gen, targets_gen, nu = nu, assume_centered = False, PATH = PATH+foldername, verbose = False) 

def exp_rdtargets(n_mc, PATH):
    pn_list = [(50,25)]
    nu = 9
    K = 10
    cov_gen = cov_gen_Kblocks(10, coeff= 11 - np.ones(10).cumsum())
    
    verbose = False
    try:
        os.mkdir(PATH)
    except OSError as error:
        if verbose:
            print(error)
            
    for k in range(1,21):
        targets_gen = targets_gen_kKblocks(k, K, 0)
        foldername = "exp_rdtargets_"+str(k)
        MCs(n_mc, pn_list, cov_gen, targets_gen, nu = nu, assume_centered = False, PATH = PATH+foldername, verbose = False) 

def exp_heavy(n_mc, PATH):
    lambda_ = 2.
    pn_list = [(int(lambda_*n),n) for n in range(8,51,2)]    
    
    K=5
    k=5
    offset = 0
    verbose = False
    try:
        os.mkdir(PATH)
    except OSError as error:
        if verbose:
            print(error)
    
    coeff = (K - np.ones(K).cumsum() + 1)*2
    cov_gen = cov_gen_Kblocks(K, coeff)
    
    targets_gen = targets_gen_kKblocks(k, K, offset)
    
    for nu in [20000, 8, 4, 3, 2.5]:
        foldername = "exp_heavy_"+str(k)+"_"+str(K)+"_"+str(offset)+"_"+str(nu)    
        MCs(n_mc, pn_list, cov_gen, targets_gen, nu = nu, assume_centered = False, PATH = PATH+foldername, verbose = False) 


def exp_markowitz(PATH):
    verbose = False
    ticker_list = ['AAPL', 'ABC',     'ABMD',     'ABT',     'ADBE',     'ADI',     'ADM',     'ADP',     'ADSK',     'AEP',     'AES',     'AFL',     'AIG',     'AJG',     'ALB',     'ALK',     'ALL',     'AMAT',     'AMD',     'AME',     'AMGN',     'AON',     'AOS',     'APA',     'APD',     'APH',     'ATO',     'ATVI',     'AVB',     'AVY',     'AXP',     'AZO',     'BAC',     'BALL',     'BAX',     'BA',     'BBWI',     'BBY',     'BDX',     'BEN',     'BIIB',     'BIO',     'BKR',     'BK',     'BMY',     'BRO',     'BSX',     'BWA',     'CAG',     'CAH',     'CAT',     'CB',     'CCL',     'CDNS',     'CHD',     'CINF',     'CI',     'CLX',     'CL',     'CMA',     'CMCSA',     'CMI',     'CMS',     'CNP',     'COF',     'COO',     'COP',     'COST',     'CPB',     'CPRT',     'CPT',     'CSCO',     'CSX',     'CTAS',     'CTRA',     'CTXS',     'CVS',     'CVX',     'C',     'DD',     'DE',     'DHI',     'DHR',     'DISH',     'DIS',     'DLTR',     'DOV',     'DRE',     'DRI',     'DTE',     'DUK',     'DVA',     'DVN',     'DXC',     'D',     'EA',     'ECL',     'ED',     'EFX',     'EIX',     'EL',     'EMN',     'EMR',     'EOG',     'EQR',     'ESS',     'ES',     'ETN',     'ETR',     'EVRG',     'EXC',     'EXPD',     'FAST',     'FCX',     'FDX',     'FISV',     'FITB',     'FMC',     'FRT',     'F',     'GD',     'GE',     'GILD',     'GIS',     'GLW',     'GL',     'GPC',     'GWW',     'HAL',     'HAS',     'HBAN',     'HD',     'HES',     'HIG',     'HOLX',     'HON',     'HPQ',     'HRL',     'HSIC',     'HST',     'HSY',     'HUM',     'IBM',     'IDXX',     'IEX',     'IFF',     'INCY',     'INTC',     'INTU',     'IPG',     'IP',     'ITW',     'IT',     'IVZ',     'JBHT',     'JCI',     'JKHY',     'JNJ',     'JPM',     'J',     'KEY',     'KIM',     'KLAC',     'KMB',     'KO',     'KR',     'K',     'LEN',     'LHX',     'LH',     'LIN',     'LLY',     'LMT',     'LNC',     'LNT',     'LOW',     'LRCX',     'LUMN',     'LUV',     'L',     'MAA',     'MAS',     'MCD',     'MCHP',     'MCK',     'MCO',     'MDT',     'MGM',     'MHK',     'MKC',     'MLM',     'MMC',     'MMM',     'MNST',     'MOS',     'MO',     'MRK',     'MRO',     'MSFT',     'MSI',     'MS',     'MTB',     'MTCH',     'MU',     'NDSN',     'NEE',     'NEM',     'NI',     'NKE',     'NLOK',     'NOC',     'NSC',     'NTAP',     'NTRS',     'NUE',     'NVR',     'NWL',     'ODFL',     'OKE',     'OMC',     'ORCL',     'ORLY',     'OXY',     'O',     'PAYX',     'PCAR',     'PEAK',     'PEG',     'PENN',     'PEP',     'PFE',     'PGR',     'PG',     'PHM',     'PH',     'PKI',     'PNC',     'PNR',     'PNW',     'POOL',     'PPG',     'PPL',     'PSA',     'PTC',     'PVH',     'QCOM',     'RCL',     'REGN',     'REG',     'RE',     'RF',     'RHI',     'RJF',     'RMD',     'ROK',     'ROL',     'ROP',     'ROST',     'RTX',     'SBUX',     'SCHW',     'SEE',     'SHW',     'SIVB',     'SJM',     'SLB',     'SNA',     'SNPS',     'SO',     'SPGI',     'SPG',     'STE',     'STT',     'STZ',     'SWKS',     'SWK',     'SYK',     'SYY',     'TAP',     'TECH',     'TER',     'TFC',     'TFX',     'TGT',     'TJX',     'TMO',     'TRMB',     'TROW',     'TRV',     'TSCO',     'TSN',     'TT',     'TXN',     'TXT',     'TYL',     'T',     'UDR',     'UHS',     'UNH',     'UNP',     'USB',     'VFC',     'VLO',     'VMC',     'VNO',     'VRTX',     'VTRS',     'VZ',     'WAB',     'WAT',     'WBA',     'WDC',     'WEC',     'WELL',     'WFC',     'WHR',     'WMB',     'WMT',     'WM',     'WRB',     'WST',     'WY',     'XEL',     'XOM',     'XRAY',     'ZBRA',     'ZION']    
    domain_list, domains = get_domain_list(ticker_list)    
    domain_list = np.array(domain_list)    
    p = len(ticker_list)
    X = np.zeros((0,len(ticker_list)))
    for year in range(2010, 2022):
        X = np.concatenate([X,close_SP500(year, ticker_list=ticker_list, verbose = True)], axis=0)
    Y = np.log(X[1:]) - np.log(X[:-1])
    
    cluster_targets = np.zeros((p,p,11))
    last_target = np.eye(p)
    for i in range(11):
        idx = domain_list == i
        idxs = idx[:,np.newaxis] * idx[np.newaxis,:]
        target = np.zeros((p,p))
        target += idxs*np.eye(p)
        cluster_targets[:,:,i] = target
    targets = np.concatenate([last_target[:,:,np.newaxis], cluster_targets], axis=2)
    tilde_targets, free_indices, P = gram_schmidt(targets)
    
    targets_lancewicki = np.concatenate([last_target[:,:,None], cluster_targets], axis=2)

    p = Y.shape[1]
    
    rst_tab = []
    time_tab = []
    # lag is the number of months used to fit the estimator covariance
    lag_list = [3, 4, 6, 9, 12, 15]
    for lag in lag_list:
        print("\tMonths used to fit the estimators:", lag)
        res_s = []
        res_t = []
        for i in range(lag,Y.shape[0]//20):
            rst_t += [[]]
            Y_train = Y[max(0, (i-lag)*20):i*20]
            Y_test = Y[i*20:(i+1)*20]
            
            beg = time.perf_counter()
            S = EmpiricalCovariance(store_precision=False, assume_centered=False).fit(Y_train).covariance_
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_MTSE = MTSE_estimator(Y_train, tilde_targets, assume_centered=False, assume_orthogonal=True)
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_MTSE_L = MTSE_Lancewicki(Y_train, targets_lancewicki, assume_centered=False)
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_MTSE_G = MTSE_Gray(Y_train, targets_lancewicki, assume_centered=False)
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_LWO = LWO_estimator(Y_train, assume_centered=False, S_r=None)
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_an = analytical_shrinkage.AS_estimator(Y_train, assume_centered = False)
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_GIS = GIS.GIS(Y_train, assume_centered = False)
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_QIS = QIS.QIS(Y_train, assume_centered = False)
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_LIS = LIS.LIS(Y_train, assume_centered = False)
            end = time.perf_counter()
            rst_t[-1] += [end-beg]

            beg = time.perf_counter()
            S_OAS = OAS(store_precision=False, assume_centered=False).fit(Y_train).covariance_
            end = time.perf_counter()
            rst_t[-1] += [end-beg]
            
            res_s += [[]]
            res_e += [[]]
            for Sigma in [S, S_MTSE, S_MTSE_L, S_MTSE_G, S_LWO, S_an, S_GIS, S_QIS, S_LIS, S_OAS]:
                e, s, IC = GMV(Y_test, Sigma, wb=np.ones(p)/p)
                res_e[-1] += [e]
        
        res_s = np.array(res_s)
        res_t = np.array(res_t)
        
        algos = ["S", "MTSE", "MTSE_L", "MTSE_G", "LWO", "ANS", "GIS", "QIS", "LIS", "OAS"]
        rst_tab += [[]]
        time_tab += [[]]
        for i in range(res_s.shape[1]):
            print("\t\t",algos[i],"cumulative variance/time: \t",res_s[:,i].sum(), "\t|\t", res_t[:,i].mean())
            rst_tab[-1] += [res_s[:,i].sum()]
            time_tab[-1] += [res_t[i]]
        
    rst_tab = np.array(rst_tab)
    time_tab = np.array(time_tab)
    
    
    filename = "markowitz.pkl"
    try:
        os.mkdir(PATH)
    except OSError as error:
        if verbose:
            print(error)
    path = os.path.join(PATH, filename)
    
    dic = {
        'lag_list':lag_list,
        'cum_variance': rst_tab,
        'time': time_tab
        }
    with open(path, 'wb') as file:
        pickle.dump(dic, file)

    return rst_tab, time_tab

if __name__ == "__main__":
    PATH = ".\\rst\\" # folder where you will store the results
    # Make sure that the folder is empty before launching the experiments
    # Particularly, launching again the experiments without cleaning the folder in between makes the plots crash.
        
    # target usefulness experiment 
    # due to high number of draws (n_mc), this experiment may take several hours to compute
    print("Experiment of target usefulness: current Time =", dt.datetime.now().strftime("%H:%M:%S"))
    foldername = "exp_usefulness\\"
    n_mc = 50000
    exp_usefulness(n_mc, PATH+foldername)
    plot_usefulness(PATH=PATH+foldername)
    
    # experiment of adding random/useless targets to a good set of targets 
    # due to high number of draws (n_mc), this experiment may take several hours to compute
    print("Experiment of random/useless targets: current Time =", dt.datetime.now().strftime("%H:%M:%S"))
    foldername = "exp_rdtargets\\"
    n_mc = 5000
    exp_rdtargets(n_mc, PATH+foldername)
    plot_rdtargets(PATH=PATH+foldername)
    
    # convergence under heavy tails experiment
    # due to high number of draws (n_mc), this experiment may take several hours/days to compute
    print("Experiment of convergence under heavy tails: current Time =", dt.datetime.now().strftime("%H:%M:%S"))
    foldername = "exp_heavy\\"
    n_mc = 20000
    exp_heavy(n_mc, PATH+foldername)
    plot_heavy(PATH=PATH+foldername)
        
    # Markowitz General Minimum Variance portfolio experiment
    # due to data loading and number of months, this experiment may take several hours to compute
    print("Experiment of Markowitz GMV with real data: current Time =", dt.datetime.now().strftime("%H:%M:%S"))
    foldername = "GMV"
    rst_tab = exp_markowitz(PATH+foldername)