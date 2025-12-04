"""
Module d'entraînement et de ré-entraînement du modèle CNN
Gère la fusion des données (Sign Language + Feedback) et le fine-tuning
"""
import os
import shutil
import numpy as np
from PIL import Image
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow n'est pas installé. L'entraînement sera simulé.")

# Try to import kagglehub
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("⚠️  kagglehub n'est pas installé. Impossible de télécharger le dataset.")

class ModelTrainer:
    """Gère le ré-entraînement du modèle"""
    
    def __init__(self, db_manager, model_dir='model', input_size=(64, 64), classes=None):
        self.db_manager = db_manager
        self.model_dir = model_dir
        self.input_size = input_size
        # Classes par défaut (35 classes: 1-9, A-Z)
        default_classes = (
            '1,2,3,4,5,6,7,8,9,'
            'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,'
            'P,Q,R,S,T,U,V,W,X,Y,Z'
        ).split(',')
        self.classes = classes or default_classes
        
        # Mapping des classes (label -> index)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def retrain(self) -> Dict[str, Any]:
        """
        Lance le processus de ré-entraînement complet
        
        Returns:
            Dict avec le statut et les infos
        """
        # 1. Créer le log d'entraînement
        run_id = self.db_manager.create_training_run()
        print(f"🚀 Démarrage de l'entraînement (Run ID: {run_id})")
        
        try:
            if not TF_AVAILABLE:
                return self._simulate_training(run_id, "TensorFlow manquant")
            
            if not KAGGLEHUB_AVAILABLE:
                return self._simulate_training(run_id, "kagglehub manquant")

            # 2. Télécharger/Charger le dataset Kaggle
            print("📦 Téléchargement/Vérification du dataset Sign Language...")
            dataset_path = kagglehub.dataset_download("harshvardhan21/sign-language-detection-using-images")
            dataset_path = Path(dataset_path)
            print(f"   Dataset localisé: {dataset_path}")
            
            # Trouver le dossier contenant les images
            data_dir = self._find_data_dir(dataset_path)
            if not data_dir:
                raise FileNotFoundError("Impossible de trouver le dossier d'images dans le dataset téléchargé")
            
            print(f"   Dossier images: {data_dir}")

            # 3. Préparer les datasets (Train/Val)
            print("🔄 Préparation des datasets...")
            batch_size = 32
            img_size = self.input_size
            
            # Utiliser image_dataset_from_directory
            train_ds = tf.keras.utils.image_dataset_from_directory(
                str(data_dir),
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=img_size,
                batch_size=batch_size,
                label_mode='int' # Les labels seront des entiers correspondant à l'ordre alphabétique des dossiers
            )
            
            val_ds = tf.keras.utils.image_dataset_from_directory(
                str(data_dir),
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=img_size,
                batch_size=batch_size,
                label_mode='int'
            )
            
            # Vérifier que les classes correspondent
            dataset_classes = train_ds.class_names
            print(f"   Classes trouvées dans le dataset: {len(dataset_classes)}")
            # Note: On suppose ici que les classes du dataset correspondent à self.classes
            # Idéalement, il faudrait faire un mapping si elles diffèrent.
            
            # Optimisation
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

            # 4. Intégrer les feedbacks
            print("🔍 Récupération des feedbacks utilisateurs...")
            feedback_data = self.db_manager.get_feedback_data_for_training()
            print(f"   {len(feedback_data)} images de feedback trouvées")
            
            if feedback_data:
                try:
                    feedback_images = []
                    feedback_labels = []
                    
                    for item in feedback_data:
                        try:
                            img_path = item['image_path']
                            true_label = item['user_feedback']['true_label']
                            
                            if true_label not in self.class_to_idx:
                                print(f"   ⚠️ Label inconnu ignoré: {true_label}")
                                continue
                                
                            if os.path.exists(img_path):
                                # Charger et prétraiter l'image
                                img = Image.open(img_path)
                                # Convertir en RGB si nécessaire (le modèle attend 3 canaux)
                                img = img.convert('RGB')
                                img = img.resize(self.input_size)
                                img_arr = np.array(img).astype('float32') 
                                # Note: Rescaling layer dans le modèle fera la division / 255.
                                
                                feedback_images.append(img_arr)
                                feedback_labels.append(self.class_to_idx[true_label])
                        except Exception as e:
                            print(f"   Erreur image feedback: {e}")
                            
                    if feedback_images:
                        print(f"   ✅ {len(feedback_images)} images de feedback valides ajoutées")
                        
                        # Créer un dataset TensorFlow pour les feedbacks
                        feedback_ds = tf.data.Dataset.from_tensor_slices((
                            np.array(feedback_images),
                            np.array(feedback_labels)
                        ))
                        
                        # Batcher comme le dataset principal
                        feedback_ds = feedback_ds.shuffle(len(feedback_images)).batch(batch_size)
                        
                        # Fusionner avec le dataset d'entraînement
                        train_ds = train_ds.concatenate(feedback_ds)
                        
                        # Re-mélanger le tout
                        train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
                        
                except Exception as e:
                    print(f"⚠️ Erreur lors de l'intégration des feedbacks: {e}")
            else:
                print("ℹ️  Aucun feedback à intégrer")
            
            # 5. Construire le modèle
            print("🧠 Construction du modèle CNN...")
            num_classes = len(dataset_classes)
            
            model = keras.Sequential([
                keras.layers.Input(shape=(*img_size, 3)),
                keras.layers.Rescaling(1./255),
                
                # Augmentation de données intégrée
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                keras.layers.RandomZoom(0.1),
                
                # Bloc 1
                keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                
                # Bloc 2
                keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                
                # Bloc 3
                keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                
                # Dense
                keras.layers.Flatten(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(num_classes, activation="softmax")
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            
            # 6. Entraîner avec callback de progression
            print("🔥 Lancement de l'entraînement (5 époques)...")
            
            # Créer un callback pour mettre à jour la progression
            class ProgressCallback(keras.callbacks.Callback):
                def __init__(self, db_manager, run_id, total_epochs):
                    super().__init__()
                    self.db_manager = db_manager
                    self.run_id = run_id
                    self.total_epochs = total_epochs
                    
                def on_epoch_end(self, epoch, logs=None):
                    # Calculer le pourcentage de progression (époque terminée + 1)
                    progress = ((epoch + 1) / self.total_epochs) * 100
                    message = f"Époque {epoch + 1}/{self.total_epochs} - Accuracy: {logs.get('accuracy', 0):.2%}"
                    
                    self.db_manager.update_training_progress(
                        run_id=self.run_id,
                        progress=progress,
                        current_epoch=epoch + 1,
                        total_epochs=self.total_epochs,
                        message=message
                    )
                    print(f"   Progression mise à jour: {progress:.1f}%")
            
            total_epochs = 5
            progress_callback = ProgressCallback(self.db_manager, run_id, total_epochs)
            
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=total_epochs,
                verbose=1,
                callbacks=[progress_callback]
            )
            
            # 7. Sauvegarder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_model_name = f"cnn_model_{timestamp}.h5"
            new_model_path = os.path.join(self.model_dir, new_model_name)
            
            os.makedirs(self.model_dir, exist_ok=True)
            model.save(new_model_path)
            print(f"✅ Nouveau modèle sauvegardé: {new_model_path}")
            
            # Copier vers le modèle par défaut
            main_model_path = os.path.join(self.model_dir, 'sign_language_cnn.h5')
            try:
                shutil.copy2(new_model_path, main_model_path)
                print(f"✅ Modèle principal mis à jour: {main_model_path}")
            except Exception as e:
                print(f"⚠️ Erreur copie modèle principal: {e}")
            
            # 8. Update log
            self.db_manager.update_training_run(
                run_id=run_id,
                status="success",
                model_path=new_model_path,
                used_feedback_count=0
            )
            
            return {
                "status": "success", 
                "message": "Modèle ré-entraîné avec succès sur le dataset Sign Language!",
                "model_path": new_model_path
            }
            
        except Exception as e:
            print(f"❌ Erreur critique entraînement: {e}")
            import traceback
            traceback.print_exc()
            self.db_manager.update_training_run(
                run_id=run_id,
                status="failed",
                error_message=str(e)
            )
            return {"status": "error", "message": str(e)}

    def _simulate_training(self, run_id, reason):
        """Simule un entraînement si les dépendances manquent"""
        import time
        print(f"⚠️  Simulation d'entraînement ({reason})...")
        time.sleep(2)
        new_model_name = f"cnn_model_simulated_{datetime.now().strftime('%Y%m%d%H%M%S')}.h5"
        self.db_manager.update_training_run(
            run_id=run_id,
            status="success",
            model_path=os.path.join(self.model_dir, new_model_name),
            used_feedback_count=0
        )
        return {"status": "success", "message": f"Entraînement simulé ({reason})"}

    def _find_data_dir(self, base_path: Path) -> Path:
        """Cherche récursivement le dossier contenant les images"""
        # Stratégie: chercher un dossier qui contient des sous-dossiers (classes)
        # qui eux-mêmes contiennent des images.
        for p in base_path.rglob('*'):
            if p.is_dir():
                subdirs = [d for d in p.iterdir() if d.is_dir()]
                if len(subdirs) > 1: # Au moins 2 classes
                    # Vérifier si le premier sous-dossier contient des images
                    first_class = subdirs[0]
                    has_images = any(first_class.glob('*.jpg')) or any(first_class.glob('*.png'))
                    if has_images:
                        return p
        return None
