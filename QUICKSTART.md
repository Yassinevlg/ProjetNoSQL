# 🚀 Guide de Démarrage Rapide - CNN + MongoDB

## Étapes Essentielles

### 1️⃣ Installer MongoDB

Vous avez deux options:

#### Option A: MongoDB Local (Recommandé pour développement)

1. Télécharger MongoDB Community Server:
   - 🔗 https://www.mongodb.com/try/download/community
   - Choisir: Windows, Version 7.0+, MSI

2. Installer avec les paramètres par défaut
   - ✅ Cocher "Install MongoDB as a Service"
   - ✅ Cocher "Install MongoDB Compass" (GUI optionnelle)

3. Vérifier l'installation:
   ```powershell
   mongod --version
   ```

#### Option B: MongoDB Atlas (Cloud - Gratuit)

1. Créer un compte: https://www.mongodb.com/cloud/atlas/register
2. Créer un cluster gratuit (M0)
3. Créer un utilisateur database
4. Ajouter votre IP à la whitelist (ou autoriser 0.0.0.0/0 pour test)
5. Récupérer l'URI de connexion
6. Mettre à jour `.env`:
   ```env
   MONGO_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/
   ```

### 2️⃣ Installer les Dépendances Python

```powershell
# Dans le dossier du projet
python -m pip install -r requirements.txt
```

⏱️ Cette commande prend ~5-10 minutes (TensorFlow est volumineux)

### 3️⃣ Lancer l'Application

```powershell
python app.py
```

Vous devriez voir:
```
🚀 Démarrage de l'application CNN + MongoDB
🔄 Chargement du modèle...
⚠️  Modèle de démonstration créé!
✅ Application prête!
🌐 Accédez à l'application sur: http://localhost:5000
```

### 4️⃣ Tester l'Application

1. Ouvrir http://localhost:5000 dans votre navigateur
2. Cliquer sur "Commencer une Prédiction"
3. Téléverser une image (n'importe laquelle pour tester)
4. Voir le résultat
5. Donner un feedback
6. Consulter l'Historique et les Statistiques

## 🔧 Dépannage

### ❌ Erreur: "ModuleNotFoundError: No module named 'flask'"
→ Les dépendances ne sont pas installées
```powershell
python -m pip install -r requirements.txt
```

### ❌ Erreur: "pymongo.errors.ServerSelectionTimeoutError"
→ MongoDB n'est pas accessible
- Si MongoDB local: vérifier que le service est démarré
- Si Atlas: vérifier l'URI et la whitelist IP

### ❌ L'application démarre mais le modèle ne charge pas
→ C'est normal! L'application crée un modèle de démo automatiquement
→ Les prédictions seront aléatoires (c'est pour tester l'interface)

### 📝 Ajouter un Vrai Modèle

Si vous avez un modèle CNN entraîné (`.h5`):
```powershell
# Copier votre modèle
Copy-Item votre_modele.h5 model\cnn_model.h5

# Adapter la configuration dans .env
# MODEL_CLASSES=vos,classes,ici
```

## 📊 Fonctionnalités à Démontrer

### Pour la Démo/Rapport:

1. **Architecture Flask**
   - Routes RESTful
   - Templates Jinja2
   - Gestion des fichiers uploadés

2. **Intégration CNN**
   - Prétraitement d'images
   - Prédictions avec probabilities
   - Confiance du modèle

3. **MongoDB NoSQL**
   - Insertion de documents
   - Requêtes flexibles
   - Agrégations avancées ($group, $match, $avg)
   - Mise à jour de feedbacks

4. **Interface Utilisateur**
   - Design moderne responsive
   - Drag & drop upload
   - Visualisations interactives
   - Graphiques de statistiques

## 🎯 Points Clés pour le Rapport

### Technologies Utilisées:
- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML**: TensorFlow/Keras (CNN)
- **Database**: MongoDB (PyMongo)
- **Design**: Modern CSS avec animations

### Opérations MongoDB Implémentées:

**CRUD:**
- `insert_one()` - Sauvegarder prédictions
- `find()`, `find_one()` - Récupérer données
- `update_one()` - Mettre à jour feedbacks
- `delete_one()` - Supprimer prédictions

**Agrégations:**
```python
# Exemple dans db_manager.py
pipeline = [
    {"$group": {
        "_id": "$predicted_label",
        "count": {"$sum": 1},
        "avg_confidence": {"$avg": "$confidence"}
    }},
    {"$sort": {"count": -1}}
]
```

### Avantages NoSQL Démontrés:
- ✅ Schéma flexible (ajout de champs facile)
- ✅ Documents imbriqués (user_feedback, meta)
- ✅ Agrégations puissantes
- ✅ Scalabilité horizontale
- ✅ Requêtes rapides avec indexes

## 📸 Captures pour le Rapport

Prendre des screenshots de:
1. Page d'accueil avec statistiques
2. Upload d'image (drag & drop)
3. Résultat de prédiction avec probabilités
4. Historique des prédictions
5. Page statistiques avec graphiques
6. MongoDB Compass montrant les documents

## ✅ Checklist Avant la Démo

- [ ] MongoDB est démarré/accessible
- [ ] Dépendances Python installées
- [ ] Application lance sans erreurs
- [ ] Au moins 5-10 prédictions test effectuées
- [ ] Feedbacks donnés sur quelques prédictions
- [ ] Page statistiques affiche des graphiques
- [ ] Captures d'écran prises
- [ ] Rapport rédigé

## 🎓 Concepts à Expliquer

1. **Pourquoi MongoDB?**
   - NoSQL pour flexibilité
   - JSON-like documents naturels pour ML
   - Agrégations pour analytics

2. **Architecture de l'App**
   - MVC pattern
   - Séparation des concerns
   - Configuration centralisée

3. **Workflow Complet**
   - Upload → CNN → MongoDB → Visualisation
   - Boucle de feedback pour amélioration

---

**Bonne chance! 🍀**
