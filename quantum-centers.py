import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import dimod
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
import neal


print(__doc__)

# #############################################################################
# Generate sample data
X, labels_true = make_blobs(
    n_samples=10, cluster_std=0.5, random_state=0
)


a = 10
b = 20
c = 30

h = {}
J = {(a,b): 0.5, (a,c): 0.01, (b,c): 0.25}

exact_sampler = dimod.ExactSolver()
bqm = dimod.BinaryQuadraticModel.from_ising(h,J)

response = exact_sampler.sample(bqm)
print(response)

