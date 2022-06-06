#                  HADDAD Hassiba
#                  ML.21 : classifiaction 


#              HADDAD Hassiba 
#             SKlearn 


import numpy as np                                                            
from sklearn.datasets import make_regression 
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR
import matplotlib.pyplot as plt   
import pandas as pd 
import seaborn as sns 
from sklearn.neighbors import KNeighborsClassifier




titanic=sns.load_dataset('titanic')
dim=titanic.shape
a=titanic.head()
print(dim)
print()
print(a)



titanic=titanic[['survived', 'pclass', 'sex', 'age' ]]
titanic.dropna(axis=0, inplace=True)
titanic['sex'].replace(['male', 'female'], [0,1], inplace=True )
b=titanic.head()
print(b)



model=KNeighborsClassifier()
y=titanic['survived']
X=titanic.drop('survived' , axis=1)


model.fit(X, y)
model.score(X, y)
model.predict(X)


def survie(model, pclass=3 , sex=0, age=26):
    x=np.array([pclass, sex, age]).reshape(1,3)
    print(model.predict(x))

survie(model)










