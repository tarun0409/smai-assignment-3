import numpy as np
import pandas as pd
import random
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

class KMeans:
    centroids = None
    n_clusters = 5
    def fit(self, X, n_clusters):
        
        self.centroids = dict()
        self.n_clusters = n_clusters
        random.seed(1000)

        for i in range(0,n_clusters):
            curr_index = random.randint(0,len(X)-1)
            self.centroids[i] = X.iloc[curr_index].values

        clusters = [0]*X.shape[0]
        update_occurred = True 
        while update_occurred:
            update_occurred = False
            cluster_sum = dict()
            cluster_cardinality = dict()
            for i in range(0,n_clusters):
                cluster_sum[i] = np.zeros((1,X.shape[1]))
                cluster_cardinality[i] = 0
            index = 0
            for rand_index,row in X.iterrows():
                x = row.values
                best_cluster = -1
                best_distance = -1
                for i in range(0,n_clusters):
                    c = self.centroids[i]
                    dist = np.sum(np.square(np.subtract(c,x)))
                    if best_distance < 0 or dist < best_distance:
                        best_distance = dist
                        best_cluster = i
                if best_cluster != clusters[index]:
                    clusters[index] = best_cluster
                    update_occurred = True
                curr_cluster = clusters[index]
                cluster_sum[curr_cluster] += row.values.reshape(1,X.shape[1])
                cluster_cardinality[curr_cluster] += 1
                index += 1
            for i in range(0,n_clusters):
                self.centroids[i] = cluster_sum[i]/cluster_cardinality[i]

    def predict(self, X):
        y_predict = list()
        for index,row in X.iterrows():
            x = row.values
            best_cluster = -1
            best_distance = -1
            for i in range(0,self.n_clusters):
                c = self.centroids[i]
                dist = np.sum(np.square(np.subtract(c,x)))
                if best_distance < 0 or dist < best_distance:
                    best_distance = dist
                    best_cluster = i
            y_predict.append(best_cluster)
        return y_predict
    
    def compute_purity(self, y_predict, y_actual):
        correct = 0
        for i in range(0,len(y_actual)):
            if y_actual[i] == y_predict[i]:
                correct += 1
        purity = float(correct)/float(len(y_actual))
        return purity

def compute_purity(y_train, y_train_predict, y_actual, y_predict, y_label):
    
    cluster_label_map = dict()
    unique, counts = np.unique(y_train_predict, return_counts=True)
    cluster_dict = dict(zip(unique, counts))
    class_dict = dict()
    y_list = y_train[y_label].tolist()
    for i in y_train[y_label].unique():
        class_dict[i] = y_list.count(i)
    while cluster_dict:
        cluster = max(cluster_dict,key=cluster_dict.get)
        clas = max(class_dict,key=class_dict.get)
        cluster_label_map[cluster] = clas
        del cluster_dict[cluster]
        del class_dict[clas]
    y_pred = map(lambda x : cluster_label_map[x], y_predict)
    y_act = y_actual[y_label].tolist()
    
    correct = 0
    for i in range(0,len(y_act)):
        if y_act[i] == y_pred[i]:
            correct += 1
    purity = float(correct)/float(len(y_act))
    return purity

data = pd.read_csv("intrusion_data.csv")
pca = PCA()
X = data[['duration', 'service', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_access_files', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']].copy()

k = pca.get_best_k(X,0.1)
X_reduced = pca.reduce_dimensions(X,k)

cols = list()
for i in range(0,k):
    cols.append('A'+str(i+1))

data_reduced = pd.DataFrame(X_reduced)
data_reduced.columns = cols
data_reduced['xAttack'] = data['xAttack'].tolist()

X_train, X_valid, y_train, y_valid = train_test_split(
    data_reduced[cols],
    data_reduced[['xAttack']],
    test_size=0.3,
    random_state=0)

test_file_name = sys.argv[1]
data_test = pd.read_csv(test_file_name)
X_test = data_test[['duration', 'service', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_access_files', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']].copy()
y_test = data_test[['xAttack']]
data_test_reduced = pca.reduce_dimensions(X_test,k)
X_test_reduced = pd.DataFrame(data_test_reduced)


kms = KMeans()
kms.fit(X_train, 5)
y_pred_tr_kms = list(kms.predict(X_train))
y_pred_vd_kms = list(kms.predict(X_valid))


train_purity = compute_purity(y_train, y_pred_tr_kms, y_train, y_pred_tr_kms, 'xAttack')
valid_purity = compute_purity(y_train, y_pred_tr_kms, y_valid, y_pred_vd_kms, 'xAttack')

y_pred_ts_kms = list(kms.predict(X_test_reduced))
test_purity = compute_purity(y_train, y_pred_tr_kms, y_test, y_pred_ts_kms, 'xAttack')

print '******************** K-Means Clustering result ********************************'
print 'Train data set purity : '+str(round(train_purity*100,2))+'%'
print 'Test data set purity : '+str(round(valid_purity*100,2))+'%'
print 'Input data set purity : '+str(round(test_purity*100,2))+'%'
print '************************************************************************'