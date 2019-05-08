import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import sys

class PCA:
    
    def get_best_k(self, X, error_percent):
        for col in X:
            mean = X[col].mean()
            std = X[col].std()
            X[col] = (X[col] - mean)/std
#             X[col] = (X[col]-mean)/std
        max_cols = X.shape[1]
        for K in range(1,max_cols):
            U,S,V = svds(X,k=K)
            X_approx = np.dot(U,np.dot(np.diag(S),V))
            new_s = np.sum(np.sum(np.square(np.subtract(X,X_approx))))
            old_s = np.sum(np.sum(np.square(X)))
            diff = float(new_s)/float(old_s)
            if diff < error_percent:
                return K
    
    def reduce_dimensions(self, X, K):
        U, S, V = svds(X, k=K)
        return U

data = pd.read_csv("intrusion_data.csv")

pca = PCA()
X = data[['duration', 'service', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_access_files', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']].copy()

k = pca.get_best_k(X,0.1)
print 'Training set Best K : '+str(k)
X_reduced = pca.reduce_dimensions(X,k)

data_reduced = pd.DataFrame(X_reduced)
data_reduced['xAttack'] = data['xAttack'].tolist()

print 'Data after reduced dimensions using PCA: '
print
print
print data_reduced.head()

test_file_name = sys.argv[1]
data_test = pd.read_csv(test_file_name)
X_test = data_test[['duration', 'service', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_access_files', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']].copy()
k = pca.get_best_k(X_test,0.1)
print 'Test file set Best K : '+str(k)
X_test_reduced = pca.reduce_dimensions(X_test,k)
data_test_reduced = pd.DataFrame(X_test_reduced)
data_test_reduced['xAttack'] = data_test['xAttack'].tolist()

print 'Test file set data after reduced dimensions using PCA: '
print
print
print data_test_reduced.head()