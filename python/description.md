# Reproduce the results

There are two python files, my_methods.py and main.py.
The implementation of my methods for the competition is in the my_methods.py.

For convenience, the total 13 datasets and corresponding top-1 results are already in the experiments and results folder respectively.

To reproduce the results, you can simply run the main.py, the results will be stored in the results folder.  

There are some comments in the code to help to understand,
and for more details about the code, can see below. 


# Details about the code

There are two methods used totally in the final results, which I named Pruning_VAR and Stacking_VAR respectively. 

Here, I will explain some thoughts behind the code and share my views about why it works.

# Pruning_VAR

The First thing I want to share is, even though the Vector Autoregression or VAR is a method for linear data, VAR also is a powerful model to judge whether there is a link between variables for non-linear data.

In my opinion, with large sample sizes, VAR has a good approximation for the non-linear relation of variables.
But the coefficient may not good enough, we can't simply use the coefficient as the final scores_matrix.
The good thing is the goal of this competition is to find out whether there is a causal link i to j, which should be 0 or 1 instead of a coefficient value.
Therefore, we can use information from p-value to modify the coefficient.
And for variables i and j, if the p-value is low and the coefficient is high, then we can make the value of scores_matrix[i,j] higher.

The implementation of the Pruning_VAR can be found in the my_methods.py .
There are a few lines may be confusing in the code and too much to comment, I would explain more about it here.

(1)

```
data -= data.mean(axis=0)
```
here, we didn't standardize data but just subtract the mean value of data for the reason that dividing by the std of data may shrink the effect between two variables.

During the competition, the experiment results show that it really works well and better than standardizing data or doing nothing.
Maybe there are some better data preprocessing ways that could be tried.

(2)
```
if( p_matrix[i, j] < pvalue_threshold):
    if( val_matrix[i,j] > score_threshold):
        val_matrix[i, j] = max(1 - p_matrix[i,j], val_matrix[i,j]) + args.alpha
    else:
        val_matrix[i, j] = np.abs(val_matrix[i,j] + args.beta) 
```
There are 4 parameters for this method, but from my perspectives, the key parameters are pvalue_threshold and score_threshold.
For the pvalue_threshold and score_threshold, I believe that a value of causal link i to j should be higher if the p-value is low(means the result is reliable) and the score is high(means there is a link from i to j).

The default values of args.alpha and args.beta should work well and don't need to adjust too much. 
By the way, it's not intuitive why the args.beta parameter works, but in the experiment, it also helps to improve the result. 
In my guess, the AUC metrics maybe prefer higher values in some cases.

From the experiments, this method works for many datasets such as FinalWEATH, FinalWEATHnoise, FinalWEATHmiss and even FinalWEATHsub. 

# Stacking_VAR

The idea of Stacking_VAR is simple that the results are different with different maxlag for the VAR method and maybe we can stack these results together to get a better result.

Since the goal of this competition is to find out whether there is a causal link i to j, We prefer a higher (e.g. 1) value instead of coefficient in some cases. So in the implementation, we choose to stack the results of the Pruning_VAR method with different maxlag.

From the experiments, I found this method works better than the Pruning_VAR for 'FinalWEATHnoise_N-10_T-1000' and 'FinalWEATHnoise_N-10_T-2000' datasets.

