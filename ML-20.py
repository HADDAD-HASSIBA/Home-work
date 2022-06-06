#              HADDAD Hassiba 
#             SKlearn 


import numpy as np                                                            
from sklearn.datasets import make_regression 
from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR
import matplotlib.pyplot as plt   




# modele de regression lineaire :
  
np.random.seed(0)
m=100
X=np.linspace(0, 10, m).reshape(m,1)
y=X+np.random.rand(m, 1)

model=LinearRegression()
model.fit(X,y)                          #entrainer sur le model de x et y
model.score(X, y)                       # c'est coeficent de détermination 
model.predict(X)                        # pour faire de nouvel pédection sur les valeur de X et y 

predictions=model.predict(X) 
plt.scatter(X,y)
plt.plot(X, predictions , c='r' )

# ce modele, on le peut pas l'utilise dans toute les situations non lineaire comme x**2, donc il faut trouver un autre modele qu'on verra juste aprés 






# erreur dans un modele non lineaire

np.random.seed(0)
m=100
X=np.linspace(0, 10, m).reshape(m,1)
y=X**2 + np.random.rand(m, 1)

model=LinearRegression()
model.fit(X,y)                         
model.score(X, y)                       
model.predict(X)                      

predictions=model.predict(X) 
plt.scatter(X,y)
plt.plot(X, predictions , c='r' )





# modele SVR
np.random.seed(0)
m=100
X=np.linspace(0, 10, m).reshape(m,1)
y=X**2 + np.random.rand(m, 1)

model=SVR(kernel='rbf', degree=2, gamma='scale', coef0=0.0, tol=0.001, C=100, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
model.fit(X,y)                         
model.score(X, y)                       
model.predict(X)                      

predictions=model.predict(X) 
plt.scatter(X,y)
plt.plot(X, predictions , c='r' )




