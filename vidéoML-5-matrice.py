#                    HADDAD Hassiba 
#                    vidéo ML5 : les matrices



# premiere mot pour utiliser les matrices il nous faut un librairie qui s'appelle numpy


import numpy as np                                                             # pour l'instaler : tools- pythonpath manager- add path - utilisateur- hassiba-anaconda- lib- site_package


#définir une matrice :
matrice=np.array([[1, 2], [3,4],[5,6]])                                        # nous avons crée  une matrice, ce qui entre les crochets sont des ligne et ce qui est dans les crochet sont les colones.
dimension=matrice.shape                                                        # on utilise le shape pour  connaitre les dimension de la matrice
print(" la matrice est : " , matrice, "et sa dimension est : ", dimension )


 
#========================================================================================
print()
#==========================================================================================


# pour trouver la transpose d'une matrice il faut :
transpose=matrice.T
print(" la matrice transpose est : ", transpose)
print(" la dimension de la transpose est : ", transpose.shape)


#========================================================================================
print()
#==========================================================================================

# addition des matrices :
somme=matrice+matrice
print(somme)




#========================================================================================
print()
#==========================================================================================




#matrice rempliet des zeros
zero=np.zeros((5,3))
print("affichez les elemnt de la matrice zero : ", zero)



#========================================================================================
print()
#==========================================================================================




#matrice rempliet des un :
ones=np.ones((5,3))
print("affichez les elemnt de la matrice ones : ", ones)


#========================================================================================
print()
#==========================================================================================


# le produit entre les matrices :   le nobre de colonne de la premiére egale nombre de ligne 

produit=matrice.dot(transpose)
print("le produit des deux matrices est " ,produit)
print(produit.shape)

