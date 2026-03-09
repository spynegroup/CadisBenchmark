# -*- coding: utf-8 -*-
"""
Benchmark Pgmpy cadis algos on the 'Asia' dataset.

Created on Thu Mar  5 11:28:38 2026

@author: Saptarshi Pyne
"""

## The 'Asia' dataset is taken from:
## Fig. 2, Lauritzen, Steffen L., and David J. Spiegelhalter. 
## "Local computations with probabilities on graphical structures and 
## their application to expert systems." Journal of the Royal 
## Statistical Society: Series B (Methodological) 50.2 (1988): 157-194.
##
## Number of nodes: 8
## Number of edges: 8
## Number of parameters: 18
## Average Markov blanket size: 2.5
## Average degree: 2
## Maximum in-degree: 2
##
## Each node is a binary Yes-No variable.

######################################################
## Begin: Import
######################################################

from IPython.display import Image
from pgmpy.utils import get_example_model

# from causallearn.search.ConstraintBased.PC import pc

# from castle.algorithms import PC
# import castle.algorithms

# from pgmpy.estimators import PC
import pgmpy.estimators

import numpy as np
import networkx as nx

import sklearn.metrics


######################################################
## End: Import
######################################################

## Load the 'Asia' a.k.a. 'lung cancer' network model.
## Ref: https://pgmpy.org/examples/Creating_Discrete_BN.html
asia_model = get_example_model('asia')

# type(asia_model)
#> pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork

cpds = asia_model.get_cpds()
#> [<TabularCPD representing P(asia:2) at 0x1e38adddaf0>,
#>  <TabularCPD representing P(bronc:2 | smoke:2) at 0x1e389891a60>,
#>  <TabularCPD representing P(dysp:2 | bronc:2, either:2) at 0x1e38c82a1b0>,
#>  <TabularCPD representing P(either:2 | lung:2, tub:2) at 0x1e38c829fa0>,
#>  <TabularCPD representing P(lung:2 | smoke:2) at 0x1e38c82a060>,
#>  <TabularCPD representing P(smoke:2) at 0x1e38c82a030>,
#>  <TabularCPD representing P(tub:2 | asia:2) at 0x1e38c82a0c0>,
#>  <TabularCPD representing P(xray:2 | either:2) at 0x1e38c82a0f0>]

# print(cpds[0])
# +-----------+------+
# | asia(yes) | 0.01 |
# +-----------+------+
# | asia(no)  | 0.99 |
# +-----------+------+
##
# print(cpds[1])
# +------------+------------+-----------+
# | smoke      | smoke(yes) | smoke(no) |
# +------------+------------+-----------+
# | bronc(yes) | 0.6        | 0.3       |
# +------------+------------+-----------+
# | bronc(no)  | 0.4        | 0.7       |
# +------------+------------+-----------+
##
## and so on.

## Visualize the network
viz = asia_model.to_graphviz()
viz.draw('asia.png', prog='neato')
Image('asia.png')


## There are 8 nodes
nodes = asia_model.nodes()
print(nodes)
#> ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']


## There are 8 directed edges.
## ('A', 'B') represents the edge A->B.
edges = asia_model.edges()
print(edges)
#> [('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), 
#> ('smoke', 'bronc'), ('lung', 'either'), 
#> ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')]


## Simulate a dataset with 10^3 = 1,000 samples given 
## the causal graph.
## Ref: 
## https://pgmpy.org/examples/
## Structure_Learning.html#0.-Simulate-some-sample-datasets
##
## Ref for 
## pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork.simulate():
## https://pgmpy.org/models/bayesiannetwork.html
##
## Set a seed value for reproducibility.
##
asia_data = asia_model.simulate(n_samples=int(1e3), seed=42)

# asia_data.head()
# Out[24]: 
#   asia either lung tub smoke bronc xray dysp
# 0   no    yes  yes  no   yes   yes  yes  yes
# 1   no     no   no  no    no    no   no  yes
# 2   no     no   no  no    no    no   no   no
# 3   no     no   no  no    no    no   no   no
# 4   no     no   no  no    no   yes   no  yes


asia_npa = asia_data.to_numpy()


######################################################
## Begin: Apply the PC algo on the Asia dataset
######################################################
## Run PC with default parameter values
# cg = pc(asia_npa)



######################################################
## End: Apply the PC algo on the Asia dataset
######################################################


# PC expects a pandas DataFrame with column names
# pc_castle = castle.algorithms.PC(variant='original')
# pc_castle.learn(asia_data)


## The output will be of type 'pgmpy.base.DAG.DAG'.
pc_pgmpy_est = pgmpy.estimators.PC(data=asia_data)
pc_pgmpy = pc_pgmpy_est.estimate(ci_test='chi_square', \
                               variant='stable', \
                               max_cond_vars=4, \
                               return_type='dag')

## Obtain information about the discovered causal graph
# pc_pgmpy.nodes()
#
# pc_pgmpy.edges()
#
# len(pc_pgmpy.edges())
#> 7

## Visualize the discovered grah
pc_pgmpy_viz = pc_pgmpy.to_graphviz()
pc_pgmpy_viz.draw('pc_pgmpy.png', prog='neato')
Image('pc_pgmpy.png')


## Calculate the F1-score of the predicted network with respect to the true network


# Function to evaluate the learned model structures.
def get_f1_score(estimated_model, true_model):
    nodes = estimated_model.nodes()
    
    est_adj = nx.to_numpy_array(
        estimated_model.to_undirected(), nodelist=nodes, weight=None
    )
    
    true_adj = nx.to_numpy_array(
        true_model.to_undirected(), nodelist=nodes, weight=None
    )

    f1 = sklearn.metrics.f1_score(np.ravel(true_adj), np.ravel(est_adj))
    print("F1-score for the model skeleton: ", f1)

get_f1_score(pc_pgmpy, asia_model)




