# RNN La Défense

Projet de travail collaboratif visant à tester des modèles de prédiction probabilistes basés sur des réseaux de neurones récurrents de type LSTM. Ce répertoire propose plusieurs modèles de prédiction à comparer avec les données du pôle La Défense.


# Fonctionnement
Il y a 10 modèles aux architectures différentes à comparer dans le cadre de cette étude. Il s'agit des modèles suivants : 
|  Modèle   | 
|  ----  | 
| DeepAR  | 
| LSTMIndScaling | 
| LSTMInd |
| LSTMFRScaling |
| LSTMFR | 
| LSTMCOP | 
| GPCOP | 
| GP | 
| GPScaling | 
| DeepNegPol | 

Une phase d'apprentissage est réalisée sur un jeu d'entrainement dédié puis les capacités de prédictions sont calculées sur un jeu de test. Les données sont composées de 25 séries temporelles de comptages dans le pôle de la Défense (données CML + Winflow) ainsi que des covariables calendaires ou d'événementiel (météo, concerts, offre rer, travaux,...). Différentes variantes de ces modèles devraient être comparées selon quels hyperparamètres sont pris en compte.


# Utilisation
Deux utilisateurs du projet : Renaud Oger et Paul de Nailly
Le partage des modèles testés est le suivant:
* **Renaud** : LSTMCOP, LSTMFR, LSTMFRScaling, GPCOP, GP
* **Paul** : GPScaling, LSTMIndScaling, LSTMInd, DeepAR, DeepNegPol
 
L'utilisateur devrait utiliser deux fichiers (si tout fonctionne bien !):
* hyperparams : modification des hyperparamètres à tester, on modifie 3 hyperparamètres : learning rate, nombre de cellules de la couche hidden du LSTM, nom du modèle utilisé.
* script_run : script central à exécuter. Ce script lance le préprocessing des données (division entre entrainement et test), l'entrainement avec le modèle et les paramètres choisis, le calcul de métriques de tests sur la base de test et création de 3 graphiques de prédictions sur des périodes particulières (un concert et deux perturbations).

On testera notamment les variantes de paramètres suivants : 
* **num_cells** : 30, 60, 90 
* **learning_rate** : 1e-4, 1e-3, 1e-2  (Attention ! dans le cas des modèles LSTMFR et LSTMFRScaling on modifiera le paramètre "learning_rate_fullrank" alors qu'il s'agira de "learning_rate" dans les autres cas).

Attention ! Sur spyder on redémarre un nouveau noyau entre chaque expérience afin de bien prendre en compte les nouveaux paramètres.

Les autres codes sont dataset.py (création des datasets, non utilisé ici), multivariate_models.py (création d'un dictionnaire de modèles), train_and_plot_prediction.py (ensemble de fonctions pour le préprocessing, l'entrainement et les prédictions, appelé par script_run.py). Le répertoire "model" contient l'ensemble des codes associés aux modèles. Chaque modèle contient un code "_network" permettant la mise en place de l'architecture du réseau et un code "_estimator" permettant de mettre en place les réseaux d'entrainement et de prédiction associés aux architectures.

L'utilisateur doit récupérer le répertoire (git clone) et y intégrer les données input présentes dans le répertoire \\urbanbox\edt\EDT-ETG_MOB\41Mobilite\PolitiquesTransport\2018_RIF_LissageHPM_LaDefense\Thèse Paul\Etude_Exploratoire_LSTM\1_Données_Entrée. Il modifie les hyperparamètres nécessaires dans le fichier "hyperparams" puis exécute les étapes du script "script_run.py". Les résultats sont ensuite rangés dans le répertoire \\urbanbox\edt\EDT-ETG_MOB\41Mobilite\PolitiquesTransport\2018_RIF_LissageHPM_LaDefense\Thèse Paul\Etude_Exploratoire_LSTM\2_Sorties_Modèles. L'avancement du travail est ensuite rapporté dans le fichier Synthèse_Modèles_LSTM.xlsx.
