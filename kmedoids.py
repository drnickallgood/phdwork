import matplotlib.pyplot as plt
import numpy as np

from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score


print(__doc__)

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    #n_samples=50, centers=centers, cluster_std=0.5, random_state=0
    n_samples=10, cluster_std=0.5, random_state=0
)


# #############################################################################
# Compute Kmedoids clustering

# Train Data
cobj = KMedoids(n_clusters=3).fit(X)

# Get labels from kmedioids
labels = cobj.labels_


unique_labels = set(labels)
colors = [
    plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
]

# Intrinsic methods, silohouette scores
s_score = silhouette_score(X, labels)

for k, col in zip(unique_labels, colors):

    class_member_mask = labels == k

    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.plot(
    cobj.cluster_centers_[:, 0],
    cobj.cluster_centers_[:, 1],
    "o",
    markerfacecolor="cyan",
    markeredgecolor="k",
    markersize=6,
)


'''
#plot
# plot the 3 clusters
plt.scatter(
    X[cobj == 0, 0], X[cobj == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[cobj == 1, 0], X[cobj == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[cobj == 2, 0], X[cobj == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    cobj.cluster_centers_[:, 0], cobj.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)

'''
plt.legend(scatterpoints=1)

plt.title("KMedoids clustering. Medoids are represented in cyan.")
plt.grid()
#plt.show()

#print("Silhouette Score: ", s_score)

#Inertia is our objective function, lower is better, 
print("Objective function: ", cobj.inertia_)
print("Cluster Center Coordinates: ", cobj.cluster_centers_)

'''
{
"id": [ ID ],
"action": "parse_lib_and_version",
"libs":[  {"numpy-0.00", "matplotlib-0.00", "pandas-0.00" }   ],
}

{
"id": [ ID ],
"action": "create_virtual_env", "pip_install_libs"
"libs":[  {"numpy-0.00", "matplotlib-0.00", "pandas-0.00" }   ],
}

{
"id": [ ID ],
"action":  "pip_install_libs"
"libs":[  {"numpy-0.00", "matplotlib-0.00", "pandas-0.00" }   ],
}

{
"id": [ ID ],
"action": "install_radare2",
"libs":[  {"radare2" }   ],
}

{
"id": [ ID ],
"action": "install_radare2",
"libs":[  {"radare2" }   ],
}


{
"id": [ ID ],
"action":  "venv_load_libs"
"libs":[  {"numpy-0.00", "matplotlib-0.00", "pandas-0.00", "radare2","dis" }   ],
}

{
"id": [ ID ],
"action":  "create_call_graphs_venv"
"libs":[  ],
}

{
"id": [ ID ],
"action":  "quantum_formulate"
"libs":[  ],
}

{
"id": [ ID ],
"action":  "quantum_send"
"libs":[  ],
}

'''

