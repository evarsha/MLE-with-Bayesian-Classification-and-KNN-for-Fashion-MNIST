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


###BAYES CLASSIFIER FOR Train SET ---- BUILT IN



clf = GaussianNB()
clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)

train_acc = accuracy_score(y_train,y_train_pred)*100
print("Train accuracy of Bayes Classifier: ",train_acc)


# In[7]:


###BAYES CLASSIFIER FOR TEST SET ---- BUILT IN



clf = GaussianNB()
clf.fit(X_train,y_train)

y_test_pred = clf.predict(X_test)

test_acc = accuracy_score(y_test,y_test_pred)*100
print("Test accuracy of Bayes Classifier: ",test_acc)


# In[8]:


###   K-NN CLASSIFIER FOR TRAIN SET ---- BUILT IN



#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(X_train,y_train)

#KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

#y_train_pred = knn.predict(X_train)

#train_acc = accuracy_score(y_train,y_train_pred)*100

#print("Train accuracy of K-NN classifier for K=3: ",train_acc)


# In[9]:


###   K-NN CLASSIFIER FOR TEST SET ---- BUILT IN



knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

y_test_pred = knn.predict(X_test)

test_acc = accuracy_score(y_test,y_test_pred)*100

print("Test accuracy of K-NN classifier for K=3: ",test_acc)


# In[10]:


### PCA

pca = PCA(n_components=50)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)


print("Shape of Feature Test set Before PCA: ", X_test.shape)
print("Shape of Feature Test set After PCA: ", X_test_pca.shape)


###   K-NN CLASSIFIER FOR TEST SET 



knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca,y_train)

KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

y_test_pred = knn.predict(X_test_pca)

test_acc_pca = accuracy_score(y_test,y_test_pred)*100

print("Test accuracy of K-NN classifier for K=3 after PCA: ",test_acc_pca)


# In[11]:


###  LDA


lda = LinearDiscriminantAnalysis()

X_train_lda = lda.fit_transform(X_train,y_train)

X_test_lda = lda.transform(X_test)


print("Shape of Feature Test set Before LDA: ", X_test.shape)
print("Shape of Feature Test set After LDA: ", X_test_lda.shape)



###   K-NN CLASSIFIER FOR TEST SET 


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_lda,y_train)

KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

y_test_pred = knn.predict(X_test_lda)

test_acc_lda = accuracy_score(y_test,y_test_pred)*100

print("Test accuracy of K-NN classifier for K=3 after LDA: ",test_acc_lda)




# In[ ]:




