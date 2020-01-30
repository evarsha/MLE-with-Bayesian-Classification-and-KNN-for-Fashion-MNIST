#!/usr/bin/env python
# coding: utf-8

# ###LOADING LIBRARIES

# In[1]:



import numpy as np

import os
import gzip

import mnist_reader

import matplotlib.pyplot as plt


# ###LOADING BUILT-IN FUNCTIONS

# In[2]:



from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score


from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[3]:


###LOADING MNIST DATASET


def load_mnist(path, kind='train'):
   
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)

    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)

    print("Dataset Loaded")
    
    return images, labels


# In[4]:


###LOADING TRAIN AND TEST SET FEATURES AND LABELS



X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


# In[5]:


###NORMALIZING AND CHECKING THE SHAPES OF TRAIN AND TEST SETS



X_train = X_train/255

X_test = X_test/255

print("Feature Train and Test datasets are normalized")

#print(X_train[1])

#X_train = X_train.reshape(X_train.shape).T   

#y_train = y_train[np.newaxis]

#X_test = X_test.reshape(X_test.shape).T   

#y_test = y_test[np.newaxis]

print("Shape of Train set features (X_train) :  ",X_train.shape)
print("Shape of Train set labels (y_train) :  ",y_train.shape)
print("Shape of Test set features (X_test) :  ",X_test.shape)
print("Shape of Test set labels (y_test) :  ",y_test.shape)


# In[6]:


###  SORTING THE FEATURES AND LABELS


X_train_ord=[]
y_train_ord=[]


for i in range(0,max(y_train)+1):
    for j in range(0,len(y_train)):
        if y_train[j] == i:
            
            X_train_ord.append(X_train[j])
            y_train_ord.append(y_train[j])
                    
X_train = np.asarray(X_train_ord)
y_train = np.asarray(y_train_ord)

print(X_train[59999][783])
#print(y_train[0:5])
print(X_train.shape[0])
#print(X_train[1])
#A = np.cov(X_train)


# In[7]:


###  SORTING THE FEATURES AND LABELS


X_test_ord=[]
y_test_ord=[]

for i in range(0,max(y_test)+1):
    for j in range(0,len(y_test)):
        if y_test[j] == i:
            
            X_test_ord.append(X_test[j])
            y_test_ord.append(y_test[j])
                    
X_test = np.asarray(X_test_ord)
y_test = np.asarray(y_test_ord)


print(len(X_test))


# In[8]:


### DEFINING MEAN AND COVARIANCE MATRIX:

def mean_and_cov(X_train):
    
    mean = 0
    sum = 0
    mean_mat = []
    cov = []
    tot_cov = []
    count = 0

    
    for i in range(0,max(y_train)+1):
   
        for j in range(0,X_train.shape[0]):
           
            if y_train[j]==i:
                
                cov.append(X_train[j])
                sum = sum + X_train[j]
                count = count + 1
            
        mean = sum/count
        covar_mat = np.cov(np.asarray(cov).T) + 0.6*np.identity(X_train.shape[1])
        cov = []
        count = 0    
        mean_mat.append(mean)
        tot_cov.append(covar_mat)
    
    mean_mat = np.asarray(mean_mat)
    print((np.asarray(tot_cov)).shape)
    print(mean_mat)
    print(mean_mat.shape)
    
    return mean, mean_mat, tot_cov


# In[9]:


mean, mean_mat, tot_cov = mean_and_cov(X_train)


# In[10]:


def max_likelihood(features, mean_mat, tot_cov):
    
    
    numen = np.subtract(features, mean_mat)
   
    denom = np.matmul(numen.T, np.linalg.inv(tot_cov))
    
    denom = np.dot(denom,numen) 
   
    mle = -(denom + np.log(np.linalg.det(tot_cov)))/2
    
    return mle
  


# In[11]:


b = X_test[i].reshape(X_test[i].shape[0],1)
print(b.shape)


# In[12]:


mle_test = 0
mle_pred = []
j=0

for i in range(0, len(X_test)):
    for cls in range(0,max(y_test)+1):
        X_test2 = X_test[i].reshape(X_test[i].shape[0],1)
        mean_mat2 = mean_mat[cls].reshape(mean_mat[cls].shape[0],1)
        tot_cov2 = tot_cov[cls].reshape(tot_cov[cls].shape[0],tot_cov[cls].shape[0])
        #print(X_test2.shape)
        #print(mean_mat[cls].shape)
        #print(tot_cov[cls].shape)
        mle_test_current = max_likelihood(X_test2, mean_mat2, tot_cov2)
        #print(mle_test_current)
        
        if mle_test_current > mle_test:
            mle_test_current = mle_test
            mle_pred_current = cls
            
    mle_pred.append(mle_pred_current)
    if i%500==0:
        print(i)
            


# In[20]:


## BAYES CLASSIFIER FOR Train SET ---- BUILT IN



clf = GaussianNB()
clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train,y_train_pred)*100
print("Train accuracy of Bayes Classifier: ",train_acc)


# In[21]:


###BAYES CLASSIFIER FOR TEST SET ---- BUILT IN



clf = GaussianNB()
clf.fit(X_train,y_train)

y_test_pred = clf.predict(X_test)

test_acc = accuracy_score(y_test,y_test_pred)*100
print("Test accuracy of Bayes Classifier: ",test_acc)


# In[22]:


### PCA

pca = PCA(n_components=50)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)


print("Shape of Feature Test set Before PCA: ", X_test.shape)
print("Shape of Feature Test set After PCA: ", X_test_pca.shape)


###BAYES CLASSIFIER FOR PCA TEST SET 



clf = GaussianNB()
clf.fit(X_train_pca,y_train)

y_test_pred = clf.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_test_pred)*100
print("Test accuracy of Bayes Classifier After PCA: ",test_acc_pca)


# In[23]:


###  LDA


lda = LinearDiscriminantAnalysis()

X_train_lda = lda.fit_transform(X_train,y_train)

X_test_lda = lda.transform(X_test)


print("Shape of Feature Test set Before LDA: ", X_test.shape)
print("Shape of Feature Test set After LDA: ", X_test_lda.shape)


###BAYES CLASSIFIER FOR LDA TEST SET



clf = GaussianNB()
clf.fit(X_train_lda,y_train)

y_test_pred = clf.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_test_pred)*100
print("Test accuracy of Bayes Classifier After LDA: ",test_acc_lda)


