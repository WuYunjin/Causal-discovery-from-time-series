
# this script used for saving all tried methods
# Imports
import json
import zipfile
import bz2
import time
import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
import networkx as nx
import cdt
from cdt.causality.graph import LiNGAM
from cdt.causality.graph import MMPC
from cdt.causality.graph import GES

def LiNGAM_model(data):
       # Input data is of shape (time, variables)
    T, N = data.shape
        # Standardize data
    # data -= data.mean(axis=0)
    # data /= data.std(axis=0)
    # data = pd.DataFrame(data)
    data = pd.DataFrame(data)
    obj = LiNGAM()    
    output = obj.predict(data)
    matrix= nx.adjacency_matrix(output).todense()
    val_matrix = np.array(np.abs(matrix))

    p_matrix = None
    # Matrix of time lags
    lag_matrix = None
    return val_matrix, p_matrix, lag_matrix

def Lasso_GES(data):
    T, N = data.shape
    data = pd.DataFrame(data)

    glasso = cdt.independence.graph.Glasso()
    skeleton = glasso.predict(data)
    new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne') #稀疏化
    model = cdt.causality.graph.GES()
    output_graph = model.predict(data, new_skeleton)
    matrix = nx.adjacency_matrix(output_graph).todense()

    val_matrix = np.array(np.abs(matrix))

    p_matrix = None
    # Matrix of time lags
    lag_matrix = None
    return val_matrix, p_matrix, lag_matrix

def CAM_model(data):
    data = pd.DataFrame(data)
    data = pd.DataFrame(data)
    obj = cdt.causality.graph.CAM()   
    output = obj.predict(data)
    matrix= nx.adjacency_matrix(output).todense()
    val_matrix = np.array(np.abs(matrix))

    p_matrix = None
    # Matrix of time lags
    lag_matrix = None
    return val_matrix, p_matrix, lag_matrix

def GES_model(data):
       # Input data is of shape (time, variables)
    T, N = data.shape
        # Standardize data
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    data = pd.DataFrame(data)
    data = pd.DataFrame(data)
    obj = GES()    
    output = obj.predict(data)
    matrix= nx.adjacency_matrix(output).todense()
    val_matrix = np.array(np.abs(matrix))

    p_matrix = None
    # Matrix of time lags
    lag_matrix = None
    return val_matrix, p_matrix, lag_matrix


def LiNGAM_VAR(data,maxlags=2):

   # Input data is of shape (time, variables)
    T, N = data.shape

    # Standardize data
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Fit VAR model and get coefficients and p-values
    tsamodel = tsa.var.var_model.VAR(data)
    results = tsamodel.fit(maxlags=maxlags,  trend='nc')

    pvalues = results.pvalues
    values = results.coefs # VAR结果的系数
    noise = results.resid #VAR结果的残差
    val_matrix = np.zeros((N, N), dtype='float32')

    #LiNGAM python包要求data是pd.DataFrame()格式
    noise = pd.DataFrame(noise)
    lingam_model = LiNGAM()
    output = lingam_model.predict(noise)
    B0 = nx.adjacency_matrix(output).todense()
    B0 = B0.T #对LiNGAM返回的结果做转置
    #print("B0:",B0)
    B_i = np.zeros((maxlags,N, N), dtype='float32')
    M = values
    tmp = (np.eye(N)-B0)
    for i in range(maxlags):        
        B_i[i] =  np.dot(tmp, M[i])
        #print("B{}:{}".format(i+1,B_i[i]))
    
    #利用B0更新一下val_matrix。
    B0 = np.abs(B0.T)

    for j in range(N):
        for i in range(N):

            # Store only values at lag with minimum p-value
            tau_min_pval = np.argmin(pvalues[
                                    (np.arange(1, maxlags+1)-1)*N + i , j]) + 1

            # Store absolute coefficient value as score
            val_matrix[i, j] = np.abs(B_i[tau_min_pval-1, j, i])

            #取系数更大的那个 
            #if(val_matrix[i, j]< B0[i,j]):
           #     val_matrix[i, j] = B0[i,j]
  
    

    #print("val_matrix:",val_matrix)
    # Matrix of p-values
    p_matrix = None

    # Matrix of time lags
    lag_matrix = None
    return val_matrix, p_matrix, lag_matrix


def linear_VAR(data, maxlags=2, correct_pvalues=True):
    # this is VAR model

    # Input data is of shape (time, variables)
    T, N = data.shape

    # Standardize data
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Fit VAR model and get coefficients and p-values
    tsamodel = tsa.var.var_model.VAR(data)
    results = tsamodel.fit(maxlags=maxlags,  trend='nc')
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
                                    (np.arange(1, maxlags+1)-1)*N + i , j]) + 1
            p_matrix[i, j] = pvalues[(tau_min_pval-1)*N + i , j]

            # Store absolute coefficient value as score
            val_matrix[i, j] = np.abs(values[tau_min_pval-1, j, i])

            # Store lag
            lag_matrix[i, j] = tau_min_pval

    if correct_pvalues:
        p_matrix *= float(maxlags) 
        p_matrix[p_matrix > 1.] = 1.
    #print(type(val_matrix))
    return val_matrix, p_matrix, lag_matrix

#-----------------------------------------------------------------------------------------------------------------
#the below is for running experiments

def run_experiment(dataname,num_N,num_T,method,parameter_maxlags=2):

    # Setup a python dictionary to store method hash, parameter values, and results
    results = {}

    ################################################
    # Identify method and used parameters
    ################################################

    # Method name just for file saving
    method_name = method

    # Insert method hash obtained from CauseMe after method registration
    method_hash = {"linear-VAR":"7c262d386a154d9f90feea42af3f2b86","LiNGAM-VAR":"5c05b7d2b8d5429abba3e396796a9041","LiNGAM": "aea69a5b895c4120828317cc367b2aed",
                    "GES":"a2168aed82884a3ea324641657993603","Lasso_GES":"dd36e4e0369540c48f6e23d150a8c58f","CAM":"f530c87507b44ab38f9889d3a88032b3"}
    results['method_sha'] = method_hash[method_name]

    # The only parameter here is the maximum time lag
    maxlags = parameter_maxlags

    results['parameter_values'] = "maxlags=%d" % maxlags

    #################################################
    # Experiment details
    #################################################
    # Choose model and experiment as downloaded from causeme
    results['model'] = dataname #'TestCLIMnoise'

    # Here we choose the setup with N=3 variables and time series length T=150
    experimental_setup =  'N-{}_T-{}'.format(num_N,num_T) #'N-20_T-5000'
    results['experiment'] = results['model'] + '_' + experimental_setup

    # Adjust save name if needed
    save_name = '{}_{}_{}'.format(method_name,
                                  results['parameter_values'],
                                  results['experiment'])

    # Setup directories (adjust to your needs)
    experiment_zip = 'experiments/%s.zip' % results['experiment']
    results_file = 'results/%s.json.bz2' % (save_name)

    #################################################

  # Start of script
    scores = []
    pvalues = []
    lags = []
    runtimes = []

    # (Note that runtimes on causeme are only shown for validated results, this is more for
    # your own assessment here)

    # Loop over all datasets within an experiment
    # Important note: The datasets need to be stored in the order of their filename
    # extensions, hence they are sorted here
    print("Load data")
    with zipfile.ZipFile(experiment_zip, "r") as zip_ref:
        for name in sorted(zip_ref.namelist()):

            print("Run {} on {}".format(method_name, name))
            data = np.loadtxt(zip_ref.open(name))
            ######################
            import seaborn as sns
            import pandas as pd
            import matplotlib.pyplot as plt
            sns.pairplot(pd.DataFrame(data))
            plt.show()
            continue
            ######################

            # Runtimes for your own assessment
            start_time = time.time()

            # Run your method (adapt parameters if needed)
            if(method_name=="linear-VAR"): val_matrix, p_matrix, lag_matrix = linear_VAR(data)
            elif(method_name=="LiNGAM-VAR"): val_matrix, p_matrix, lag_matrix = LiNGAM_VAR(data)
            elif(method_name=="LiNGAM"): val_matrix, p_matrix, lag_matrix = LiNGAM_model(data)
            elif(method_name=="GES"): val_matrix, p_matrix, lag_matrix = GES_model(data)
            elif(method_name=="Lasso_GES"): val_matrix, p_matrix, lag_matrix = Lasso_GES(data)
            elif(method_name=="CAM"): val_matrix, p_matrix, lag_matrix = CAM_model(data)
            else: print("method doesn't exist")
            runtimes.append(time.time() - start_time)
            print(val_matrix)
            # Now we convert the matrices to the required format
            # and write the results file
            scores.append(val_matrix.flatten())
            # pvalues and lags are recommended for a more comprehensive method evaluation,
            # but not required. Then you can leave the dictionary field empty          
            if p_matrix is not None: pvalues.append(p_matrix.flatten())
            if lag_matrix is not None: lags.append(lag_matrix.flatten())

    # Store arrays as lists for json
    results['scores'] = np.array(scores).tolist()
    if len(pvalues) > 0: results['pvalues'] = np.array(pvalues).tolist()
    if len(lags) > 0: results['lags'] = np.array(lags).tolist()
    results['runtimes'] = np.array(runtimes).tolist()
    print(len(results['scores']))

    # Save data
    print('Writing results ...')
    results_json = bytes(json.dumps(results), encoding='latin1')
    with bz2.BZ2File(results_file, 'w') as mybz2:
        mybz2.write(results_json)

if __name__ == '__main__':

    for method in ["LiNGAM-VAR"]:
        for data in ["TestWEATHnoise"]: 
            run_experiment(data,5,2000,method)
