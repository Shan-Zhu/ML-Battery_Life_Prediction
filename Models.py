
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd


# input data
data_train=pd.read_csv('data-2-1.csv',sep=',') #or 'data-100-1.csv'

labels=data_train['Type'][:,np.newaxis]
features=data_train.drop('Type', axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(features, labels, test_size=0.3, random_state=0)#


names = ["Decision Tree", "Nearest Neighbors","Neural Net", "AdaBoost","Gaussian Process",
         "Random Forest", "Linear SVM", "RBF SVM", "Naive Bayes"]

classifiers = [
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianProcessClassifier(),
    RandomForestClassifier(),
    SVC(),
    SVC(),
    GaussianNB()]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print (name,score)
