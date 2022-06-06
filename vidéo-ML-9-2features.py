#                    HADDAD Hassiba 
#                     vidéo ML9 : Régression  linéaire mais avec plusieurs features
 


import numpy as np                                                             
from sklearn.datasets import make_regression 
import matplotlib.pyplot as plt   
from  mpl_toolkits.mplot3d import Axes3D                                           



# dans ce progtramme au lieu de travailler avec une seule feature ( un modele linéaire), on va travailler avec 2 features dans le sens d'etudier un modele plus complexe {vous pouvez consultez la partie theorique}

# Remarque :

matrice=np.array(range(0,20, 1)).reshape(5,4)
print(matrice)

# element de matrice :
print(matrice[0,3])                                # L'element de matrice de la ligne 1 et la 3 colonne



# sous matrice :
print(matrice[1:3, 0:3])                            # une sous matrice qui contient la ligne 2 et 3 et les colonnes 1,2,3



# les lignes :
print(matrice[2,1])                                #  l'ement de matrice de 3 ligne et 2 colonne
print(matrice[0:3,])                               # la matrice qui contient seulement 3 premiere ligne et toutes les colonne
print(matrice[1:4,])                               # elle contient dela deuxiemme ligne jusqu' au 4 eme ligne 
print(matrice[1:3])                                 # print(matrice[0:3,]) 


# les colonnes : 
print(matrice[:,0:2])                               # la premiere et la deuxieme colonne
print(matrice[:,1:4])                               # toutes les lignes et 3 trois derniere colonne



# Dataset :

x, y=make_regression(n_samples=100 , n_features=2, noise=10)                   # c'est une fonction qui nous permet de générer des données mais cette fois ci le x ce n'est plus un vecteur mais c'est une matrice de 100 ligne et deux colonne x1 et x2 car nous avons choisi deux features 
#plt.scatter(x, y)                                                              #x et y doivent avoir la même taille c'est pour ca qu'on arrive pas a tracer les donnée par  scatter(x abscisse y ordonnée)
plt.scatter(x[:,0], y,s = 90, c = 'green', marker = '*')                             # cest pour tracer y en fonction des valeurs de x de la premiere colonne 
#plt.scatter(x[:,1], y,s = 90, c = 'red', marker = '+')                               # cest pour tracer y en fonction des valeurs de x de la deuxieme colonne 
plt.title('scatter plot ')
 

print("les valeurs des features x sous forme d'une matice à deux colonne sont les suivantes :")
print(x)                                                                       #on trouve que x est donnée comme une colonne de 100 ligne
print(" la dimension de vecteur x est : " , x.shape)                           #(100,1)



print("les valeurs de target y sont les suivantes :")
print(y)                              
print(" la dimension de vecteur y est : " , y.shape)                           # ah quand on affihe la dimmension de y on trouve que c'est (100, ), ce qui un peu bizzard mais ce n'est pas un buge c'est juste que la fonctioon make_regression nous donne les dim de y incomplete pour resouder ce probleme il faut faire l'etape suivante.

 


#========================================================================================================================================

print()

#========================================================================================================================================









# redimensionner les target y :


y=y.reshape((100, 1))
print(" la dimension de vecteur y aprés la modification des dim est : " , y.shape)


# ce qu'on a fait avant est juste mais ce qui est de faire c'est d'écrire un programme qu'o peut exécuter directement sans faire des chagement donc veut mieux :
y=y.reshape(y.shape[0], 1)
print(" la dimension de vecteur y aprés la modification des dim est : " , y.shape)    








#========================================================================================================================================

print()

#========================================================================================================================================











#la matrice X :

"""
d'aprés la partie théorique qu'on a fait, on a trouvé qur x est une matrice qui contient (n+1) colonne : n pour les feature  et le 1 pour les biais
mais d'aprés ce qu'on a programmé avant x est une colonne mais elle contient pas les biais, dons il faut les rajouter
"""


colonne_one=np.ones([x.shape[0],1])                              #une matrice à une colonne et le nombre de ligne est egale au nombre de ligne de la matrice x   
print(colonne_one)
X=np.hstack((x, colonne_one))                        #cette fonction nous permet de coller deux colonne au plus à une matrice

X=np.hstack((x, np.ones([x.shape[0],1]))) 
X_transpose=X.T                                       

print("la matrice qui contient le feature x et le biais est donnée par : " ,X)
print("la dimention de cette matrice est : ", X.shape)
print("la transpose de la matrice X est : ", X.T)





#========================================================================================================================================

print()

#========================================================================================================================================








# les parametre a, b, et c ou bien theta de dim (3*1):

# il faut qu'on donne au débart des valeurs aleatoire à b et a afin de corriger l'erreur 


theta=np.random.randn(3,1)

print("la matrice theta est : ")
print(theta)
print("la dimension de theta est ", theta.shape)







#========================================================================================================================================

print()

#========================================================================================================================================





# Le modele : F= X. θ   :


def modele(X, theta) :                              # il faut savoir que le modele es f=x*theta
    f=X.dot(theta)
    return f


print(modele(X, theta))       
print()

model=modele(X, theta)
print(model)
print( model.shape)


plt.scatter(x[:,0], y)                                  # afficher notre dataset
plt.plot(x[:,0], model , c='r')                         #afficher ce x en fonction de notre model






#========================================================================================================================================

print()

#========================================================================================================================================







# La fonction cout J(theta):

#elle représente l'erreur quadratique moyenne





def cost_function(X, y, theta):
    m=len(y)                                    # m c'est le nombre d'exemple qu'on a dans le dataset - question :y.shape 
    J=(1/(2*m))*np.sum((model-y)**2)
    return J



fonction_coup=cost_function(X, y, theta)

print("la valeur de la fonction coup est : " , fonction_coup)                            #elle va nous affiché des valeur tres grande mais nous on espere qu'elle tant vers 0 (l'ereur tant vers 0)








#========================================================================================================================================

print()

#========================================================================================================================================











#gradiant : J'(θ)=(1/m)*tanspose_X*(X*θ-Y)

def grad(X, theta, y):                                             #dim est 2*1
    m=len(y)
    grad_J=(1/m)*X_transpose.dot((model-y))
    return grad_J



grad=grad(X, theta, y)
print(grad)







#========================================================================================================================================

print()

#========================================================================================================================================











#descente de gradiant :  

# alpha: learning_rate   , nombre d'itération 

alpha=0.01

n_iterations= 1000



def descend_grad(X, theta, y, alpha, n_iterations) :
    for i in range(0, n_iterations) :
        theta=theta -alpha*grad
    return theta 

print(descend_grad(X, theta, y, alpha, n_iterations))         # la discente de gradiant a une dim de 2*1 aussi 



theta_finale=descend_grad(X, theta, y, alpha=0.001  , n_iterations=1000)
print(" les valeur de a et b proposé par la machine pour minimisé l'erreur : ")
print( theta_finale)                                                     # c'est la colonne theta finale qui contient les valeurs de a et b aprés la correction des ereur par la descente de gradiant







#========================================================================================================================================

print()

#========================================================================================================================================








# Machine learning :
# on remplace theta dans notre modele pour vérifier ce qui va nous donner :



prediction= modele(X, theta_finale)


print("la matrice theta : ", theta)
print("la matrice theta_finale : ", theta_finale)



plt.scatter(x[:,0], y,s = 90, c = 'green', marker = '*')  
plt.plot(x[:,0],prediction , c='r' )


plt.scatter(x[:,1], y,s = 90, c = 'green', marker = '*')  
plt.plot(x[:,1],prediction , c='r' )

#========================================================================================================================================

print()

#========================================================================================================================================



# notre dataset en 3D

fig=plt.figure()
#plt.title('les target y en fonction des deux features x1 et x2 ')
ax=fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y ,s = 50, c = 'green', marker = '*')
ax.scatter(x[:,0], x[:,1], prediction, s = 50, c = 'red', marker = '.')

# je sais qu'il ya un probleme mais je n'est pas pu le resoudre