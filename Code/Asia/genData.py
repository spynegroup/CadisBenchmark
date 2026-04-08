'''
Goal: Load the Asia model and simulate a dataset
Author: Saptarshi Pyne
Date: April 8, 2026

The 'Asia' model is taken from:
Fig. 2, Lauritzen, Steffen L., and David J. Spiegelhalter.
"Local computations with probabilities on graphical structures and
their application` to expert systems." Journal of the Royal
Statistical Society: Series B (Methodological) 50.2 (1988): 157-194.

Number of nodes: 8
Number of edges: 8
Number of parameters: 18
Average Markov blanket size: 2.5
Average degree: 2
Maximum in-degree: 2

Each node is a binary Yes-No variable.
'''

## Import modules 
import os
from pgmpy.example_models import load_model
import pickle

## Define input parameters
n_samples = 5000
seed_val = 42
outfile = os.path.normpath(os.path.join(os.getcwd(), '..', '..', 'Assets', 'Asia', 'asia_dataset.pkl'))


## class 'pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork'
asia_model = load_model("bnlearn/asia")

print(f"{len(asia_model.nodes())} Nodes            : {sorted(asia_model.nodes())}")
print(f"{len(asia_model.edges())} edges : {sorted(asia_model.edges())}")

## Get the conditional probability tables (CPTs)
cpts = asia_model.get_cpds()

## Print node-wise conditional probability tables (CPTs)
for i in cpts:
    print(i)

## Simulate a dataset.
## Returns a pandas dataframe where rows are samples and 
## columns are variables.
asia_data = asia_model.simulate(n_samples=n_samples, seed=seed_val)

print(f"Sampled {n_samples} rows — shape: {asia_data.shape}\n")

for col in asia_data.columns:
    asia_data[col] = asia_data[col].astype(str)

asia_dataset = {"asia_model": asia_model, "asia_data": asia_data}

## Write the Python objects into a binary file
with open(outfile, 'wb') as f:
    pickle.dump(asia_dataset, f)





























