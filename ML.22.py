#                  HADDAD Hassiba
#                  ML.21 : une technique pour entrainer un modele, optimiser, evaluer


#              HADDAD Hassiba 
#             SKlearn 

import numpy as np      
import matplotlib.pyplot as plt                                                     
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import confusion_matrix

from sklearn.svm import SVR
 
import pandas as pd 
import seaborn as sns 


iris=load_iris()

X=iris.data
y=iris.target
print(X.shape)
plt.scatter(X[:,0], X[:,1], c=y , alpha=0.9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

print('trainset :' , X_train.shape)
print('trainset :' , X_test.shape)




model=KNeighborsClassifier(n_neighbors= 6)
model.fit(X_train , y_train)
model.score(X_train , y_train)
print('train score : ' , model.score(X_train, y_train))


# validation set, Amélioré un modele 


cross=cross_val_score(KNeighborsClassifier(), X_train, y_train , cv=5 , scoring ='accuracy')
cross_moyenne=cross_val_score(KNeighborsClassifier(2), X_train, y_train , cv=5 , scoring ='accuracy').mean
print(cross)
print(cross_moyenne)


'''

# validation cureve     , pour detecter les over fitting
model=KNeighborsClassifier()
k = np.arange(1, 50)

train_score, val_score = validation_curve(model, X_train, y_train,  'n_neighbors' , k , cv=5)
print(val_score.mean())
plt.plot(k,val_score.mean(axis=1), lebel='validation')
plt.plot(k, train_score.mean(axis=1), lebel='train')

plt.ylabel('score')
plt.xlabel('n_neighbors')
plt.legend()

'''


# grid-search-cv

param_grid={'n_neighbors' : np.arange(1,20), 'metric' : ['euclidean' , 'manhattan']}
grid=GridSearchCV(KNeighborsClassifier() ,param_grid , cv =5 )
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_)

model.score(X_test , y_test)
confusion_matrix(y_test), model.predict(X_test)



