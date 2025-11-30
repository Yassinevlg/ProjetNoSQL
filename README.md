# Application Flask CNN + MongoDB

Une application web complète qui combine l'apprentissage profond (CNN) avec une base de données NoSQL MongoDB pour la classification d'images.

## 🎯 Objectif du Projet

Cette application permet de:
- 📤 Téléverser des images
- 🤖 Effectuer des prédictions avec un modèle CNN
- 💾 Stocker les résultats dans MongoDB
- 📊 Visualiser des statistiques et agrégations
- 💬 Collecter des feedbacks utilisateurs

## 📋 Prérequis

- Python 3.8+
- MongoDB (local ou Atlas)
- pip

## 🚀 Installation

### 1. Cloner/télécharger le projet

```powershell
cd cnn-mongodb-project
```

### 2. Installer MongoDB (si nécessaire)

**Option A: MongoDB Local**
- Télécharger: https://www.mongodb.com/try/download/community
- Installer et démarrer le service

**Option B: MongoDB Atlas (Cloud)**
- Créer un compte gratuit sur https://www.mongodb.com/cloud/atlas
- Créer un cluster
- Récupérer l'URI de connexion

### 3. Créer un environnement virtuel (recommandé)

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 4. Installer les dépendances

```powershell
pip install -r requirements.txt
```

### 5. Configuration

Copier `.env.example` en `.env` et adapter les valeurs:

```powershell
Copy-Item .env.example .env
```

Éditer `.env`:
```env
SECRET_KEY=votre_cle_secrete_aleatoire
MONGO_URI=mongodb://localhost:27017/
DATABASE_NAME=cnn_interface
MODEL_CLASSES=0,1,2,3,4,5,6,7,8,9
```

### 6. Ajouter votre modèle CNN (optionnel)

Si vous avez un modèle CNN entraîné (fichier `.h5`):
```powershell
# Placer votre modèle dans le dossier model/
Copy-Item votre_modele.h5 model/cnn_model.h5
```

**Note**: Si aucun modèle n'est fourni, l'application créera automatiquement un modèle de démonstration (non entraîné).

## ▶️ Lancement de l'Application

```powershell
python app.py
```

L'application sera accessible sur: **http://localhost:5000**

## 📁 Structure du Projet

```
cnn-mongodb-project/
│
├── app.py                      # Application Flask principale
├── config.py                   # Configuration
├── requirements.txt            # Dépendances Python
├── .env.example               # Template de configuration
├── README.md                  # Ce fichier
│
├── model/
│   ├── model_loader.py        # Chargeur du modèle CNN
│   └── cnn_model.h5          # Votre modèle (à ajouter)
│
├── utils/
│   └── db_manager.py         # Gestionnaire MongoDB
│
├── templates/                 # Templates HTML
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── history.html
│   ├── statistics.html
│   └── error.html
│
└── static/
    └── uploads/              # Images téléversées
```

## 🎨 Fonctionnalités

### 1. Page d'Accueil
- Vue d'ensemble de l'application
- Statistiques globales
- Informations sur le modèle

### 2. Prédiction
- Upload d'images (drag & drop)
- Prédiction en temps réel
- Affichage des probabilités
- Stockage automatique dans MongoDB

### 3. Historique
- Liste de toutes les prédictions
- Indicateurs de feedback
- Filtrage et recherche

### 4. Statistiques (MongoDB Agrégations)
- **Précision globale** du modèle
- **Distribution par classe** prédite
- **Distribution des vraies classes** (feedbacks)
- **Données de confusion**
- Graphiques interactifs

## 💾 Opérations MongoDB

L'application utilise plusieurs opérations MongoDB:

### CRUD de Base
- **Create**: `insert_one()` pour sauvegarder les prédictions
- **Read**: `find()`, `find_one()` pour récupérer les données
- **Update**: `update_one()` pour les feedbacks
- **Delete**: `delete_one()`, `delete_many()` pour supprimer

### Agrégations
- `$group` - Regrouper par classe
- `$match` - Filtrer les documents
- `$sort` - Trier les résultats
- `$avg` - Calculer les moyennes
- `$sum` - Compter les occurrences
- `$project` - Formater les résultats
- `$cond` - Conditions dans les agrégations

## 🔧 Configuration Avancée

### Personnaliser le Modèle

Dans `.env`, adapter selon votre modèle:

```env
# Pour MNIST (chiffres 0-9)
MODEL_CLASSES=0,1,2,3,4,5,6,7,8,9

# Pour des lettres
MODEL_CLASSES=A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z

# Pour des catégories personnalisées
MODEL_CLASSES=chat,chien,oiseau
```

Dans `config.py`, adapter la taille d'entrée:

```python
MODEL_INPUT_SIZE = (28, 28)    # Pour MNIST
MODEL_INPUT_SIZE = (224, 224)  # Pour ResNet, VGG, etc.

MODEL_GRAYSCALE = True   # Pour images en niveaux de gris
MODEL_GRAYSCALE = False  # Pour images couleur (RGB)
```

## 📊 API Endpoints

L'application expose aussi des endpoints JSON:

- `GET /api/stats` - Statistiques globales en JSON
- `GET /api/predictions/recent?limit=10` - Dernières prédictions

Exemple:
```bash
curl http://localhost:5000/api/stats
```

## 🐛 Dépannage

### Erreur de connexion MongoDB
```
MongoClient cannot connect to mongodb://localhost:27017/
```
→ Vérifier que MongoDB est démarré

### Le modèle ne charge pas
```
Le fichier du modèle n'existe pas
```
→ Placer votre fichier `.h5` dans `model/cnn_model.h5`  
→ Ou laisser l'application créer un modèle de démo

### Erreur d'import TensorFlow
```
No module named 'tensorflow'
```
→ Réinstaller: `pip install tensorflow==2.15.0`

## 📝 Pour la Démo / Rapport

### Points à Démontrer

1. **Flask Routes**
   - Route `/predict` pour upload et prédiction
   - Route `/history` pour l'historique
   - Route `/statistics` pour les agrégations

2. **Intégration CNN**
   - Chargement du modèle
   - Prétraitement des images
   - Prédictions avec probabilities

3. **MongoDB Operations**
   - Insert de nouvelles prédictions
   - Queries pour récupérer l'historique
   - Updates pour les feedbacks
   - Agrégations pour les statistiques

4. **Interface Utilisateur**
   - Design moderne et responsive
   - Visualisations interactives
   - Feedback utilisateur

### Scénario de Test

1. Démarrer l'application
2. Naviguer vers **Prédiction**
3. Upload une image
4. Voir le résultat et la confiance
5. Donner un feedback (correct/incorrect)
6. Consulter l'**Historique**
7. Voir les **Statistiques** MongoDB

## 🎓 Concepts NoSQL Illustrés

- **Documents flexibles** (schéma dynamique)
- **Embedded documents** (user_feedback, meta)
- **Indexes** pour optimiser les queries
- **Aggregation pipelines** pour analytics
- **Scalabilité horizontale** (MongoDB Atlas)

## 📄 Licence

Projet académique - NoSQL Course

## 👥 Auteur

Projet réalisé dans le cadre du cours NoSQL

---

**Bon courage pour votre présentation! 🚀**
