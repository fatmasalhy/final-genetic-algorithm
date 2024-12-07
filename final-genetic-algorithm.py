import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger le fichier CSV contenant les données Iris
df = pd.read_csv('iris_data.csv')
X = df.drop(columns=['target']).values
y = df['target'].values

df.head()

df.dtypes

df

df.info()

# Prétraiter les données : normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fonction pour entraîner et évaluer un modèle SVM avec des hyperparamètres donnés
def evaluate_svm(params):
    C, gamma = params
    svm = SVC(C=C, gamma=gamma)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Algorithme génétique pour l'optimisation des hyperparamètres
def genetic_algorithm(population_size, generations, mutation_rate):
    # Initialiser une population aléatoire de candidats (C, gamma)
    population = [(random.uniform(0.1, 10), random.uniform(0.001, 1)) for _ in range(population_size)]
    
    for generation in range(generations):
        print(f"Génération {generation + 1}/{generations}")
        
        # Évaluer la fitness de chaque individu (accuracy du SVM)
        fitness_scores = [evaluate_svm(individual) for individual in population]
        
        # Sélectionner les meilleurs individus (par exemple, les 50% les mieux classés)
        selected_population = [population[i] for i in np.argsort(fitness_scores)[-population_size // 2:]]
        
        # Croisement : générer de nouveaux individus en combinant les meilleurs
        offspring = []
        while len(offspring) < population_size // 2:
            parent1, parent2 = random.sample(selected_population, 2)
            crossover_point = random.randint(1, 2)
            offspring.append((parent1[0], parent2[1]) if crossover_point == 1 else (parent2[0], parent1[1]))
        
        # Mutation : appliquer une petite variation sur certains individus
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, 1)
                if mutation_point == 0:
                    offspring[i] = (offspring[i][0] * random.uniform(0.9, 1.1), offspring[i][1])
                else:
                    offspring[i] = (offspring[i][0], offspring[i][1] * random.uniform(0.9, 1.1))
        
        # La nouvelle population est composée des meilleurs individus et des descendants
        population = selected_population + offspring

    # Retourner la meilleure solution trouvée
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual, max(fitness_scores)

 # Exécuter l'algorithme génétique
best_params, best_accuracy = genetic_algorithm(population_size=10, generations=20, mutation_rate=0.1)

print(f"Meilleurs paramètres trouvés : C = {best_params[0]}, gamma = {best_params[1]}")
print(f"Exactitude du modèle avec ces paramètres : {best_accuracy}")























