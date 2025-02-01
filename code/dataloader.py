import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import datetime as dt
import os

def dataloader_YF(PATH, start_date = dt.datetime(2004, 1, 1), end_date = dt.datetime(2022, 5, 20)):
    files = os.listdir(PATH)
    dir_files = {}
    for file in files:
        ticker = file[:-4]
        df = pd.read_csv(os.path.join(PATH,file))
        # remove the beginning constant samples
        indices = (pd.to_datetime(df['Date']) > start_date) & (pd.to_datetime(df['Date']) < end_date)
        df = df[indices]
        # export to np arrays
        date_arr = pd.to_datetime(df['Date']).to_numpy()
        open_arr = df['Open'].to_numpy()
        close_arr = df['Close'].to_numpy()
        dir_files[ticker] = [date_arr, open_arr, close_arr]
    return dir_files

def dataloader_SP500(year, verbose = True):
    PATH = ".\\data\\YahooFinance\\SP500\\"+str(year)
    files = os.listdir(PATH)
    dir_files = {}
    for file in files:
        try:
            ticker = file[:-18]
            df = pd.read_csv(os.path.join(PATH,file))
            # export to np arrays
            date_arr = pd.to_datetime(df['Date']).to_numpy()
            open_arr = df['Open'].to_numpy()
            close_arr = df['Close'].to_numpy()
            if date_arr.shape[0] > 246:
                dir_files[ticker] = [date_arr, open_arr, close_arr]
            elif verbose:
                print(file, "is empty")
        except:
            if verbose:
                print(file, "raised an error")
    return dir_files

def close_array(dir_files, plot=False):
    ticker_list = list(dir_files.keys())
    # suppose same length per ticker
    prices = np.zeros((len(ticker_list),dir_files[ticker_list[0]][0].shape[0]))
    for i in range(len(ticker_list)):
        date_arr, open_arr, close_arr = dir_files[ticker_list[i]]
        prices[i,:close_arr.shape[0]] = close_arr
        if plot:
            plt.plot(date_arr, close_arr, label=ticker_list[i])
    if plot:
        plt.title("Close prices")
        plt.legend()
        plt.show()
    return prices

def get_ticker_list(year):
    df = dataloader_SP500(year, verbose=False)
    return list(df.keys())

def get_domain_list(ticker_list):
    PATH = ".\\data\\YahooFinance\\SP500\\domains.csv"
    df = pd.read_csv(PATH)
    domains = np.unique(df['GICS Sector'])
    domain_list = []
    for ticker in ticker_list:
        try:
            domain = df[df['Symbol'] == ticker]['GICS Sector'].values[0]
            domain_card = np.argmax(domains == domain)
        except:
            domain_card = 12
        domain_list += [domain_card]
    return domain_list, domains

def close_SP500(year, ticker_list = None, verbose = True):
    df = dataloader_SP500(year, verbose=False)
    if ticker_list == None:
        ticker_list = list(df.keys())
    d = len(ticker_list)
    N = df[ticker_list[0]][2].shape[0]
    X = np.zeros((N,d))
    indices = []
    for i in range(d):
        ticker = ticker_list[i]
        samples = df[ticker][2]
        if samples.shape[0] == N and (samples == 0).sum() == 0:
            X[:,i] = samples
            indices += [i]
        elif verbose:
            print(ticker, "unmatching size")
    indices = np.array(indices).astype(int)
    return X[:, indices]