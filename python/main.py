import json
import zipfile
import bz2
import time
import numpy as np

import my_methods
import argparse

def run_experiment(dataname,args):

    # Setup a python dictionary to store method hash, parameter values, and results
    results = {}

    ################################################
    # Identify method and used parameters
    ################################################

    # Method name just for file saving
    method_name = args.method

    # Insert method hash obtained from CauseMe after method registration
    method_hash = {"Pruning_VAR":"cc175d05b5134f5ba74f0c4e177d8ca6",
                    "Stacking_VAR":"81e27acaa7cd4045aae71e4611d12469" }
    results['method_sha'] = method_hash[method_name]

    # The only parameter here is the maximum time lag
    maxlags = args.maxlag

    results['parameter_values'] = "maxlags=%d" % maxlags

    #################################################
    # Experiment details
    #################################################
    # Choose model and experiment as downloaded from causeme
    results['model'] = dataname 

    # Here we choose the setup with N=3 variables and time series length T=150
    experimental_setup =  'N-{}_T-{}'.format(args.N,args.T) #'N-20_T-5000'
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

            # Runtimes for your own assessment
            start_time = time.time()

            # Run your method (adapt parameters if needed)
            if(method_name=="Pruning_VAR"): val_matrix, p_matrix, lag_matrix = my_methods.Pruning_VAR(data,args)
            elif(method_name=="Stacking_VAR"): val_matrix, p_matrix, lag_matrix = my_methods.Stacking_VAR(data,args)
            else: print("the method doesn't exist")
            runtimes.append(time.time() - start_time)
            #print(val_matrix)
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

    #define a parser and add the name of parameters will be used
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='input the method name, Pruning_VAR or Stacking_VAR.')
    parser.add_argument('--maxlag', type=int, help='the maxlag.')
    parser.add_argument('--N', type=int, help='the number of variables.')
    parser.add_argument('--T', type=int, help='the time length of data.')
    parser.add_argument('--Missing', type=bool, default= False,help='whether there are missing values in data.')

    parser.add_argument('--pvalue_threshold', type=float, default=0.05 ,help='the pvalue threshold for pruning.')
    parser.add_argument('--score_threshold', type=float, default=0.5 ,help='the score threshold for pruning.')

    parser.add_argument('--alpha', type=float, default=0.9,help='a constant value for pruning.')
    parser.add_argument('--beta', type=float, default=0.05,help='a constant value for pruning.')

    # set parameters and run experiments, one experiment for one dataset, totally in 13 datasets.
    args = parser.parse_args(['--method','Pruning_VAR','--N','5','--T','1000','--maxlag','9',
                            '--pvalue_threshold','0.01','--score_threshold','0.185','--alpha','0.95','--beta','0.09'])
    run_experiment('FinalWEATH',args)
    
    args = parser.parse_args(['--method','Pruning_VAR','--N','10','--T','1000','--maxlag','4',
                            '--pvalue_threshold','0.06','--score_threshold','0.20','--alpha','0.1','--beta','0.03'])
    run_experiment('FinalWEATH',args)

    args = parser.parse_args(['--method','Pruning_VAR','--N','10','--T','2000','--maxlag','7',
                            '--pvalue_threshold','0.04','--score_threshold','0.15','--alpha','0.15','--beta','0.05'])
    run_experiment('FinalWEATH',args)

    args = parser.parse_args(['--Missing','True','--method','Pruning_VAR','--N','5','--T','1000','--maxlag','16',
                           '--pvalue_threshold','0.01','--score_threshold','0.185','--alpha','0.95','--beta','0.04'])
    run_experiment('FinalWEATHmiss',args)

    args = parser.parse_args(['--Missing','True','--method','Pruning_VAR','--N','5','--T','2000','--maxlag','22',
                           '--pvalue_threshold','0.013','--score_threshold','0.185','--alpha','0.95','--beta','0.05'])
    run_experiment('FinalWEATHmiss',args)    

    args = parser.parse_args(['--Missing','True','--method','Pruning_VAR','--N','10','--T','1000','--maxlag','4',
                           '--pvalue_threshold','0.06','--score_threshold','0.20','--alpha','0.1','--beta','0.03'])
    run_experiment('FinalWEATHmiss',args)
  
    args = parser.parse_args(['--Missing','True','--method','Pruning_VAR','--N','10','--T','2000','--maxlag','7',
                           '--pvalue_threshold','0.04','--score_threshold','0.15','--alpha','0.15','--beta','0.05'])
    run_experiment('FinalWEATHmiss',args)

    args = parser.parse_args(['--method','Pruning_VAR','--N','10','--T','1000','--maxlag','6',
                            '--pvalue_threshold','0.008','--score_threshold','0.06','--alpha','0.5','--beta','0.1'])
    run_experiment('FinalWEATHsub',args)

    args = parser.parse_args(['--method','Pruning_VAR','--N','10','--T','2000','--maxlag','10',
                            '--pvalue_threshold','0.008','--score_threshold','0.08','--alpha','0.5','--beta','0.1'])
    run_experiment('FinalWEATHsub',args)

    args = parser.parse_args(['--method','Pruning_VAR','--N','5','--T','1000','--maxlag','8',
                            '--pvalue_threshold','0.0068','--score_threshold','0.027','--alpha','0.9','--beta','0.01'])
    run_experiment('FinalWEATHnoise',args)

    args = parser.parse_args(['--method','Pruning_VAR','--N','5','--T','2000','--maxlag','4',
                            '--pvalue_threshold','0.003','--score_threshold','0.035','--alpha','0.9','--beta','0.1'])
    run_experiment('FinalWEATHnoise',args)

    args = parser.parse_args(['--method','Stacking_VAR','--N','10','--T','1000','--maxlag','6',
                            '--pvalue_threshold','0.0068','--score_threshold','0.027','--alpha','0.9','--beta','0.01'])
    run_experiment('FinalWEATHnoise',args)

    args = parser.parse_args(['--method','Stacking_VAR','--N','10','--T','2000','--maxlag','7',
                            '--pvalue_threshold','0.0068','--score_threshold','0.027','--alpha','0.9','--beta','0.01'])
    run_experiment('FinalWEATHnoise',args)