import pandas as pd
import numpy as np
import operator
import json
import sys

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

data = pd.read_csv("admission_data.csv")

test_file_name = sys.argv[1]
test_data = pd.read_csv(test_file_name)

X_train, X_valid, y_train, y_valid = train_test_split(
    data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']],
    data[['Chance of Admit']],
    test_size=0.2,
    random_state=0)

X_test = test_data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y_test = test_data[['Chance of Admit']]

for col in X_train:
    mean = X_train[col].mean()
    std = X_train[col].std()
    X_train[col] = (X_train[col] - mean)/std
    X_valid[col] = (X_valid[col]-mean)/std
    y_test[col] = (X_test[col]-mean)/std

X_train['Ones'] = [1]*len(X_train)
X_valid['Ones'] = [1]*len(X_valid)
X_test['Ones'] = [1]*len(X_test)

def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-Z))

class LogisticRegression:
    
    theta = None
    threshold = 0.72
    
    def set_threshold(self,t):
        self.threshold = t
    
    def convert_to_class(self, target_list):
        return map((lambda a : 0 if a<=self.threshold else 1), target_list)
        #data['COA_logistic'] = map((lambda a : 0 if a<=0.72 else 1), (list(data['Chance of Admit'].values)))
    
    def compute_precision_recall_f1score(self, y_actual, y_predict):
        y_actual = self.convert_to_class(y_actual)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0,len(y_actual)):
            if y_actual[i]==0 and y_predict[i]==0:
                #true negative
                tn += 1
            if y_actual[i]==0 and y_predict[i]==1:
                #false positive
                fp += 1
            if y_actual[i]==1 and y_predict[i]==0:
                #false negative
                fn += 1
            if y_actual[i]==1 and y_predict[i]==1:
                #true positive
                tp += 1
        precision = float(tp)/(float(tp)+float(fp))
        recall = float(tp)/(float(tp)+float(fn))
        f1score = 2.0/((1.0/float(precision)) + (1.0/float(recall)))
        return (precision,recall,f1score)
    
    def compute_accuracy(self,y_actual, y_predict):
        y_actual = self.convert_to_class(list(y_actual))
        y_predict = list(y_predict)
        hits = 0
        for i in range(0,len(y_actual)):
            if y_actual[i] == y_predict[i]:
                hits+=1
        return float(hits)/float(len(y_actual))
    
    def predict(self, X):
        Y_pred = list(sigmoid(np.dot(X.values,self.theta.T)))
        Y_pred = self.convert_to_class(sigmoid(np.dot(X,self.theta.T)))
        return Y_pred
    
    def compute_error(self, y_pred, y_actual):
        m = len(y_pred)
        return (-1.0/float(m))*np.sum((y_actual*np.log(y_pred)) + ((1.0-y_actual)*np.log(1.0-y_pred)))
    
    def compute_gradient(self, X, h, Y):
        return np.sum(X*(h-Y), axis=0)
    
    def train(self, X_train, y_train_df, alpha, max_epochs):
        self.theta = None
        self.theta = np.random.rand(1,X_train.shape[1])
        y_train_np = y_train_df.values
        y_train_shape = y_train_np.shape
        Y = np.array(self.convert_to_class(list(y_train_np))).reshape((y_train_shape[0], 1))
        m = len(X_train)
        for i in range(0,max_epochs):
            X = X_train.values
            h = sigmoid(np.dot(X,self.theta.T))
            self.theta = self.theta - alpha*self.compute_gradient(X,h,Y)

lg = LogisticRegression()
lg.train(X_train,y_train,0.05,5000)
y_pred_train = lg.predict(X_train)
train_acc = lg.compute_accuracy(list(y_train['Chance of Admit']),y_pred_train)
train_precision,train_recall,train_f1score = lg.compute_precision_recall_f1score(list(y_train['Chance of Admit']),y_pred_train)

y_pred_valid = lg.predict(X_valid)
valid_acc = lg.compute_accuracy(list(y_valid['Chance of Admit']),y_pred_valid)
valid_precision,valid_recall,valid_f1score = lg.compute_precision_recall_f1score(list(y_valid['Chance of Admit']),y_pred_valid)

y_pred_test = lg.predict(X_test)
test_acc = lg.compute_accuracy(list(y_test['Chance of Admit']),y_pred_test)
test_precision,test_recall,test_f1score = lg.compute_precision_recall_f1score(list(y_test['Chance of Admit']),y_pred_test)

print '********************TRAINING SET*********************'
print 'ACCURACY : '+str(train_acc)
print 'PRECISION : '+str(train_precision)
print 'RECALL : '+str(train_f1score)
print '*****************************************************'
print
print '********************VALIDATION SET*********************'
print 'ACCURACY : '+str(valid_acc)
print 'PRECISION : '+str(valid_precision)
print 'RECALL : '+str(valid_f1score)
print '*****************************************************'
print
print '********************INPUT TEST SET*********************'
print 'ACCURACY : '+str(test_acc)
print 'PRECISION : '+str(test_precision)
print 'RECALL : '+str(test_f1score)
print '*****************************************************'