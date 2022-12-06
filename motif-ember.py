from ember import read_vectorized_features
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import pprint
import ember
import random

if __name__ == "__main__":

    # Get EMBER vectors for MOTIF dataset
    ember_X, y = read_vectorized_features("/media/data1/malware/MOTIF/dataset/", subset="train")

    # Normalize EMBER vectors and apply PCA
    ember_norm_X = ember_X.copy()
    ember_norm_X = make_pipeline(MinMaxScaler(), PCA(n_components=2)).fit_transform(ember_norm_X)

    num_samples = 50 
    random.seed(a=0)

    for i in range(0,num_samples):
        rand_sample = random.randint(0,num_samples)
        print(ember_norm_X[rand_sample])

