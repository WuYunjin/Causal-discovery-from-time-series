# Imports
import numpy as np
import statsmodels.tsa.api as tsa
from sklearn import preprocessing


def Pruning_VAR(data, args):
    # Input data is of shape (time, variables)
    T, N = data.shape

    if(args.Missing==True):

        impute = preprocessing.Imputer(strategy='median',missing_values=999.0)
        data = impute.fit_transform(data)

    #data processing
    data -= data.mean(axis=0)


    # Fit VAR model and get coefficients and p-values
    tsamodel = tsa.var.var_model.VAR(data)
    results = tsamodel.fit(maxlags=args.maxlag,  trend='nc')
    pvalues = results.pvalues
    values = results.coefs

    val_matrix = np.zeros((N, N), dtype='float32')

    # Matrix of p-values
    p_matrix = np.ones((N, N), dtype='float32')

    # Matrix of time lags
    lag_matrix = np.zeros((N, N), dtype='uint8')

    for j in range(N):
        for i in range(N):

            # Store only values at lag with minimum p-value
            tau_min_pval = np.argmin(pvalues[
                                    (np.arange(1, args.maxlag+1)-1)*N + i , j]) + 1
            p_matrix[i, j] = pvalues[(tau_min_pval-1)*N + i , j]

            # Store absolute coefficient value as score
            val_matrix[i, j] = np.abs(values[tau_min_pval-1, j, i])
           # val_matrix[i, j] = np.max(np.abs(values[:,j,i]))
            # Store lag
            lag_matrix[i, j] = tau_min_pval

    if 1: #always correct_pvalues
        p_matrix *= float(args.maxlag) 
        p_matrix[p_matrix > 1.] = 1.


    for j in range(N):
        for i in range(N):
            
            pvalue_threshold = args.pvalue_threshold 
            score_threshold = args.score_threshold  
            if( p_matrix[i, j] < pvalue_threshold):

                if( val_matrix[i,j] > score_threshold):
                    val_matrix[i, j] = max(1 - p_matrix[i, j],val_matrix[i,j])+args.alpha
                else:
                    val_matrix[i,j] = np.abs(val_matrix[i,j]+args.beta) 
    

    return val_matrix, p_matrix, lag_matrix


def Stacking_VAR(data,args):

    T, N = data.shape
    val_matrix = np.zeros((N, N), dtype='float32')

    # Matrix of p-values
    p_matrix = np.ones((N, N), dtype='float32')

    # Matrix of time lags
    lag_matrix = np.zeros((N, N), dtype='uint8')

    tmp_arg = args
    for lag in range(1,args.maxlag+1):

        tmp_arg.maxlag = lag
        tmp_val,tmp_p,_ = Pruning_VAR(data, tmp_arg)
        
        for j in range(N):
            for i in range(N):
                if(val_matrix[i,j] < tmp_val[i,j] or lag == 1 ):
                    p_matrix[i,j] = tmp_p[i,j] 
                    val_matrix[i,j] = tmp_val[i,j]
 
    for j in range(N):
        for i in range(N):

            pvalue_threshold = 0.1 
            score_threshold = 0.5  
            if( p_matrix[i, j] < pvalue_threshold):

                if( val_matrix[i,j] > score_threshold):
                    val_matrix[i, j] = max(1 - p_matrix[i, j],val_matrix[i,j])

    return val_matrix, p_matrix, lag_matrix