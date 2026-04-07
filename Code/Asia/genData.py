'''
Generate the Asia dataset
8 nodes
8 edges
'''

## Import modules 
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import PC, HillClimbSearch, MmhcEstimator
from pgmpy.models import DiscreteBayesianNetwork


from pgmpy.example_models import load_model
import pickle

# def load_asia_data(n_samples: int = 5000, random_state: int = 42):
n_samples = 5000
seed_val = 42
# outfile = '../../Assets/Asia/asia_dataset.pkl'
outfile = os.path.normpath(os.path.join(os.getcwd(), '..', '..', 'Assets', 'Asia', 'asia_dataset.pkl'))


## class 'pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork'
# asia_model = get_example_model("asia")
asia_model = load_model("bnlearn/alarm")

print(f"Nodes            : {sorted(asia_model.nodes())}")
print(f"True graph edges : {sorted(asia_model.edges())}")

## Get the conditional probability tables (CPTs)
cpts = asia_model.get_cpds()

## Print node-wise conditional probability tables (CPTs)
for i in cpts:
    print(i)

## Showing warning that `pgmpy.estimators.StructureScore` is deprecated 
## sampler = BayesianModelSampling(asia_model)
## data = sampler.forward_sample(size=n_samples, seed=seed_val)

## Returns a pandas dataframe where rows are samples and 
## columns are variables
asia_data = asia_model.simulate(n_samples=n_samples, seed=seed_val)

# print(f"Sampled {n_samples} rows — shape: {data.shape}\n")

for col in data.columns:
    data[col] = data[col].astype(str)

asia_dataset = {"asia_model": asia_model, "asia_data": asia_data}

## Write the Python objects into a binary file
with open(outfile, 'wb') as f:
    pickle.dump(asia_dataset, f)

# return data, asia_model




























