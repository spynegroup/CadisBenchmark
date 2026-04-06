'''
Generate the Asia dataset
'''

## Import 
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import PC, HillClimbSearch, MmhcEstimator
from pgmpy.models import DiscreteBayesianNetwork


# def load_asia_data(n_samples: int = 5000, random_state: int = 42):

asia_model = get_example_model("asia")
print(f"True graph edges : {sorted(asia_model.edges())}")
print(f"Nodes            : {sorted(asia_model.nodes())}")

# sampler = BayesianModelSampling(asia_model)
# data = sampler.forward_sample(size=n_samples, seed=random_state)
# for col in data.columns:
# data[col] = data[col].astype(str)

# print(f"Sampled {n_samples} rows — shape: {data.shape}\n")
# return data, asia_model

