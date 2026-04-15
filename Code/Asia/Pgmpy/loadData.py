'''
Goal: Load the Asia model and simulated data matrix
Author: Saptarshi Pyne
Date: April 14, 2026

'''

## Import modules 
import os
import pickle


def loadData(asia_datafile):
    ## Read/Load the dataset which is saved in a binary file.
    ## The dataset is structured as a dictionary.
    with open(asia_datafile, 'rb') as f:
        asia_dataset= pickle.load(f)

    ## Extract the Asia model.
    ## Class 'pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork'.
    asia_model = asia_dataset['asia_model']

    ## Extract the simulated Asia data.
    ## It is a pandas dataframe with 5000 rows/samples and 8 columns/variables.
    asia_data = asia_dataset['asia_data']

    ## Remove objects that are no longer required
    del asia_datafile, asia_dataset

    return asia_model, asia_data





























