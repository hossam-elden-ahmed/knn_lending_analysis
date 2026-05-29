# Now we are going to explain the neighbors method step by step
# The problem is a lending club
# The unsuccessful loan is which the funded amount by the investors falls far short of the requested loan
# The loan is to be successful if ( (loan - funded)/]loan ) >= 0.95

# ---------- Calling modules --------------------------------------------
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
from joblib.numpy_pickle_utils import xrange
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# ----------- First we call a pickle file -------------------------------
oftname  = open('dataset_small.pkl', 'rb')               # rb : a shortcut for read binary
(x, y)   = pickle.load(oftname)

# ----------- Create an instance of K-nearest neighbors classifier ------
knn   = neighbors.KNeighborsClassifier(n_neighbors=11)
knn.fit(x, y)
yhat  = knn.predict(x)
print(yhat[-1], y[-1])
print(knn.score(x,y))                             # Here we calc the accuracy of the model using score() method

# ----------- Labeling the dataset to ensure the accuracy ----------------
plt.pie(
            np.c_[np.sum(np.where(y == 1, 1, 0)),
            np.sum(np.where(y == 0, 1, 0))][0],
            labels = ['Not fully funded', 'Full amount'],
            colors =['r', 'g'],
            shadow =True,
            autopct = '%.2f')
plt.gcf().set_size_inches((7,7))
# From the pie chart we notice that the value of zeros is more ones which is imbalanced dataset

#----------- Confusion matrix -------------------------------------------
TP = np.sum(np.logical_and(yhat == 1, y == 1))
FP = np.sum(np.logical_and(yhat == 1, y == 0))
FN = np.sum(np.logical_and(yhat == 0, y == 1))
TN = np.sum(np.logical_and(yhat == 0, y == 0))
print('TN ',TN,'FP ', FP)
print('FN ',FN, 'TP ',TP)

print(metrics.confusion_matrix(y,yhat))

#----------- Overfitting --------------------------------------------------
knn  = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)
yhat = knn.predict(x)

print('Classification accuracy : '+ str(metrics.accuracy_score(yhat, y)))
print('Confusion metrix  : '+ str(metrics.confusion_matrix(yhat, y)))

# ---------- Splitting data into two subsets ------------------------------
# In this technique we split the data into two subsets PRC * 100% for training and the rest (1-PRC)*100% for testing
perm        = np.random.permutation(y.size)
PRC         = 0.7
split_point = int(np.ceil(y.shape[0])*PRC)

x_train     = x[perm[:split_point].ravel(), :]
y_train     = y[perm[:split_point].ravel()]

x_test     = x[perm[split_point:].ravel(), :]
y_test     = y[perm[split_point:].ravel()]

knn.fit(x_train, y_train)
yhat       = knn.predict(x_train)

print('\n Training stats : '.upper())
print('classification accuracy : ', str(metrics.accuracy_score(yhat, y_train)))
print('confusion metrix : \n', str(metrics.confusion_matrix(y_train, yhat)))

# Check on the test set
yhat = knn.predict(x_test)
print('training stats : '.upper())
print('classification accuracy : ', metrics.accuracy_score(yhat, y_test))
print('confusion metrix : \n', metrics.confusion_matrix(yhat, y_test))



#-------------- Splitting multiple times -------------------------------------------
# Now each time we change the point at which we split the data the accuracy changes
# So we change the point at which we split the data multiple times and calc the accuracy each time
# Then we calc the mean accuracy and that is the most approximate accuracy to this method

PRC = 0.3
acc = np.zeros((10,))
for i in xrange(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=PRC)
    knn     = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train,y_train)
    yhat    = knn.predict(x_test)
    acc[i]  = metrics.accuracy_score(yhat, y_test)
acc.shape = (1, 10)
print('Mean expected accuracy : ' + str(np.mean(acc[0])))


