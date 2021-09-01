import matplotlib.pyplot as plt
import numpy as np

from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score


print(__doc__)

# #############################################################################
# Generate sample data
X, labels_true = make_blobs(
    n_samples=10, cluster_std=0.5, random_state=0
)




