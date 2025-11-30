# Guide Complet: Entraîner le Modèle CNN sur Google Colab

## Problème
TensorFlow n'est pas disponible pour Python 3.14 sur Windows. Conda n'est pas installé sur ta machine.

## Solution
Utiliser Google Colab (gratuit, GPU inclus, pas d'installation locale) pour entraîner le modèle, puis télécharger le fichier `sign_language_cnn.h5` dans ta machine locale.

## Étapes

### 1. Ouvrir Google Colab
- Va à https://colab.research.google.com
- Clique sur "New Notebook" (ou File > New Notebook)
- Tu vas être invité à te connecter avec un compte Google (gratuit)

### 2. Copier et exécuter les cellules de code ci-dessous

**Cellule 1: Installations et imports**
```python
# Installations
!pip install kagglehub tensorflow==2.13.0 -q

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Réduire le bruit des logs TF

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

**Cellule 2: Configuration et téléchargement du dataset**
```python
# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15

# Créer un dossier temporaire pour les données
data_dir = Path('/tmp/sign_language_data')
data_dir.mkdir(exist_ok=True, parents=True)

# Télécharger le dataset depuis Kaggle
# Note: Tu dois avoir tes credentials Kaggle disponibles
# Si tu n'as pas d'API key Kaggle, va à:
# https://www.kaggle.com/settings/account -> Create New API Token
# Colab va te demander d'uploader le fichier kaggle.json

try:
    dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    print(f"Dataset téléchargé vers: {dataset_path}")
except Exception as e:
    print(f"Erreur lors du téléchargement: {e}")
    print("Assure-toi que ton API key Kaggle est configurée")
    print("File -> Upload file -> kaggle.json")
```

**Cellule 3: Créer les datasets train/val avec augmentation**
```python
# Augmentation de données
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Créer les datasets
train_dataset = keras.utils.image_dataset_from_directory(
    dataset_path,
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training"
)

val_dataset = keras.utils.image_dataset_from_directory(
    dataset_path,
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation"
)

print(f"Nombre de classes: {len(train_dataset.class_names)}")
print(f"Noms des classes: {train_dataset.class_names}")

# Normaliser les images (0-1)
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Appliquer l'augmentation au dataset d'entraînement
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
```

**Cellule 4: Construire l'architecture CNN**
```python
model = models.Sequential([
    # Bloc 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Bloc 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Bloc 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Couches denses
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_dataset.class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Cellule 5: Entraîner le modèle (⏱️ ~15-30 min)**
```python
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1
)
```

**Cellule 6: Visualiser les résultats**
```python
# Tracer l'accuracy et la loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Accuracy au fil des epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Loss au fil des epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"\nAccuracy finale sur validation: {history.history['val_accuracy'][-1]:.2%}")
```

**Cellule 7: Sauvegarder et télécharger le modèle**
```python
# Sauvegarder le modèle en format .h5
model.save('/content/sign_language_cnn.h5')
print("✓ Modèle sauvegardé à /content/sign_language_cnn.h5")

# Colab va te permettre de télécharger le fichier automatiquement
# Le fichier apparaîtra dans le panneau de fichiers à gauche
```

## 3. Télécharger le fichier sur ta machine

Après l'exécution de la **Cellule 7**:
1. Dans le panneau de fichiers Colab (à gauche), tu verras `sign_language_cnn.h5`
2. Clique sur les 3 points `⋯` à côté du fichier
3. Sélectionne "Download"
4. Le fichier va être téléchargé dans `C:\Users\yassi\Downloads\sign_language_cnn.h5`

## 4. Placer le fichier dans ton projet

```powershell
# Ouvre un terminal PowerShell et exécute:
Move-Item -Path "C:\Users\yassi\Downloads\sign_language_cnn.h5" `
          -Destination "C:\Users\yassi\.gemini\antigravity\scratch\cnn-mongodb-project\model\sign_language_cnn.h5" `
          -Force
```

Ou manuellement:
1. Navigue à `C:\Users\yassi\Downloads\`
2. Copie `sign_language_cnn.h5`
3. Ouvre `C:\Users\yassi\.gemini\antigravity\scratch\cnn-mongodb-project\model\`
4. Colle le fichier

## 5. Relancer ton application Flask

```powershell
# Dans le terminal où tu lances ton app:
cd C:\Users\yassi\.gemini\antigravity\scratch\cnn-mongodb-project
python app.py
```

L'app va automatiquement charger le modèle. Les prédictions ne seront plus aléatoires! 🎉

## Dépannage

**Q: Je dois obtenir une API key Kaggle?**
R: Oui, pour télécharger le dataset. Va à https://www.kaggle.com/settings/account et crée une API token. Colab te permettra d'uploader le fichier `kaggle.json`.

**Q: Ça prend combien de temps?**
R: ~15-30 minutes avec le GPU gratuit de Colab (beaucoup plus rapide que sur ta machine locale).

**Q: Je peux utiliser CPU au lieu du GPU?**
R: Oui, mais ce sera plus lent (~1-2 heures). GPU est recommandé (gratuit dans Colab).

**Q: Que faire si j'ai une erreur?**
R: Exécute les cellules une par une et lis les messages d'erreur attentivement. 90% des problèmes viennent de:
- API key Kaggle manquante → Va uploader kaggle.json
- Mémoire insuffisante → Réduis BATCH_SIZE à 16
- GPU non activé → Clique sur Runtime > Change runtime type > GPU

Bon entraînement! 🚀
