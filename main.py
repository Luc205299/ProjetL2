import numpy as np
import time as t
p="============================\n Bienvenue dans AI Ton Prêt \n============================\n"
strf= p.center(len(p))
print(strf)
t.sleep(0.5)
i = int(input("Saisissez l'âge de la personne : "))
t.sleep(0.25)
j = int(input("Saisissez 1 si la personne travaille et 2 si il n'en a pas : ")) #initialisation des entrées
while (j != 1 and j != 2) :
    j = int(input("Saisissez 1 si la personne travaille et 2 si il n'en a pas : "))

x_entrer = np.array(([40, 1], [35,2], [23, 2], [60,1], [36,1], [37,2], [56,2], [18,2], [21,2],[27,2],[i,j]),
                    dtype=float)  # données d'entrer avec deux conditions : age + statut de travail ( 2 = pas de travail, 1 = travail)
y = np.array(([1], [0], [1], [0], [1], [0], [1], [1], [0], [1]), dtype=float)  # données de sortie /  1 = accord /  0 = refus

# div la val max pour avoir un chiffre entre 0 et 1
x_entrer = x_entrer / np.amax(x_entrer, axis=0)  # On divise chaque entré par la valeur max des entrées

# On récupère les données
X = np.split(x_entrer, [10])[0]  # Données sur lesquelles on va s'entrainer
xPrediction = np.split(x_entrer, [10])[1]  # Valeur que l'on veut trouver


# Réseau neuronal
class Neural_Network(object):
    def __init__(self):

        #Parametre
        self.inputSize = 2  # Nombre de neurones d'entrée
        self.outputSize = 1  # Nombre de neurones de sortie
        self.hiddenSize = 3  # Nombre de neurones cachés

        #Poids
        self.P1 = np.random.randn(self.inputSize,self.hiddenSize)  # Matrice de poids entre les neurones d'entrer et cachés
        self.P2 = np.random.randn(self.hiddenSize,self.outputSize)  # Matrice de poids entre les neurones cachés et sortie

    # Fonction d'activation sigmoid, importante pour la sortie + sigmoid compris entre 0 et 1
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    # Dérivée de la fonction d'activation pour revenir en "arriere"
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # Fonction de propagation avant -> (calcul de la sortie)
    def forward(self, X):

        self.z = np.dot(X, self.P1)  # Multiplication matricielle entre les valeurs d'entrer et les poids W1
        self.z2 = self.sigmoid(self.z)  # Application de la fonction d'activation (Sigmoid)
        self.z3 = np.dot(self.z2, self.P2)  # Multiplication matricielle entre les valeurs cachés et les poids W2
        o = self.sigmoid(self.z3)  # Application de la fonction d'activation, et obtention de notre valeur de sortie final
        return o

    # Fonction de rétropropagation <- (calcul de l'erreur et ajustement des poids
    def backward(self, X, y, o):

        self.o_error = y - o  # Calcul de l'erreur
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # Application de la dérivée de la sigmoid à cette erreur

        self.z2_error = self.o_delta.dot(self.P2.T)  # Calcul de l'erreur de nos neurones cachés
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # Application de la dérivée de la sigmoid à cette erreur

        self.P1 += X.T.dot(self.z2_delta)  # On ajuste poids P1
        self.P2 += self.z2.T.dot(self.o_delta)  # On ajuste poids P2

    # Fonction d'entrainement
    def train(self, X, y):

        o = self.forward(X)
        self.backward(X, y, o)

    # Fonction de prédiction (fonction finale)
    def prediction(self):
        print("Donnée prédite apres entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))

        if (self.forward(xPrediction) < 0.5):
            print("Le pret n'est pas accordé ! \n")
        else:
            print("Le pret est accordé ! \n")


NN = Neural_Network()
print("L'IA calcule...")
for i in range(1000000):  # Choisir nombre d'itération
    NN.train(X, y)
#print final
print("Valeurs d'entrées: \n" + str(X))
print("Sortie actuelle: \n" + str(y))
print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X), 2)))
print("\n")
NN.train(X, y)
NN.prediction()
