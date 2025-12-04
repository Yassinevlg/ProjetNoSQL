# Rapport d'Analyse du Projet CNN-MongoDB

## Résumé
L'analyse approfondie du projet a révélé pourquoi les prédictions du modèle sont incorrectes. Le problème principal réside dans une incohérence critique entre la configuration de l'application, le modèle déployé et le script d'entraînement. De plus, le module de ré-entraînement automatique est inadapté au projet actuel.

## Problèmes Identifiés

### 1. Dimensions d'Entrée Incorrectes (Cause Principale des "Fausses Prédictions")
- **Configuration Actuelle** : Le fichier `config.py` définit `MODEL_INPUT_SIZE = (128, 28)` (Hauteur=128, Largeur=28).
- **Conséquence** : Cela impose un ratio d'image de **4.5:1** (une bande verticale très fine).
- **Impact** : Toutes les images envoyées au modèle sont écrasées horizontalement et étirées verticalement pour tenir dans ce format (28 pixels de large !). Cette distorsion majeure détruit les caractéristiques visuelles des signes de la main, rendant la reconnaissance impossible ou aléatoire.
- **Preuve** : Le script d'inspection du modèle a confirmé que le modèle actuel `sign_language_cnn.h5` attend bien cette forme aberrante `(128, 28, 3)`.

### 2. Incohérence avec le Notebook d'Entraînement
- Le notebook `train_sign_language_cnn.ipynb` est configuré correctement avec `IMG_SIZE = (64, 64)`, ce qui est un format carré standard pour ce type de tâche.
- **Conclusion** : Le modèle actuel dans le dossier `model/` n'a PAS été généré par ce notebook (ou alors le notebook a été modifié après coup). Il y a une divergence entre le code d'entraînement théorique et le modèle déployé.

### 3. Module de Ré-entraînement (`trainer.py`) Inutilisable
- Le fichier `model/trainer.py` contient du code "hardcodé" pour le dataset **MNIST** (reconnaissance de chiffres manuscrits 0-9 en 28x28 niveaux de gris).
- **Danger** : Si vous utilisez la fonction "Ré-entraîner" depuis l'interface Admin, le système va :
    1. Télécharger le dataset MNIST (chiffres) au lieu du langage des signes.
    2. Entraîner un modèle sur ces chiffres.
    3. Écraser votre modèle de langage des signes.
    4. Rendre l'application incapable de reconnaître les lettres (car le modèle MNIST n'a que 10 classes, alors que votre config en attend 35).

## Recommandations

Pour corriger ces problèmes et obtenir des prédictions fiables, les actions suivantes sont nécessaires :

1.  **Corriger la Configuration** : Modifier `config.py` pour utiliser une taille carrée standard, par exemple `(64, 64)` ou `(128, 128)`, cohérente avec le notebook.
2.  **Réparer le Trainer** : Réécrire `model/trainer.py` pour qu'il utilise la même logique que le notebook (téléchargement du dataset Kaggle Sign Language, architecture CNN adaptée, support RGB).
3.  **Ré-entraîner le Modèle** : Générer un nouveau modèle propre avec les bonnes dimensions.
