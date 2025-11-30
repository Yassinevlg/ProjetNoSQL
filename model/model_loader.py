"""
Chargeur et gestionnaire du modèle CNN
Gère le chargement, le prétraitement des images et les prédictions
"""
import os
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List

# Try to import TensorFlow, make it optional
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow n'est pas installe. Mode demo simple active.")



class CNNModelLoader:
    """Gestionnaire du modèle CNN pour les prédictions"""
    
    def __init__(self, model_path: str, classes: List[str], input_size: Tuple[int, int], grayscale: bool = True):
        """
        Initialise le chargeur de modèle
        
        Args:
            model_path: Chemin vers le fichier du modèle (.h5)
            classes: Liste des classes (labels) du modèle
            input_size: Taille d'entrée (largeur, hauteur)
            grayscale: True si le modèle attend des images en niveaux de gris
        """
        self.model_path = model_path
        self.classes = classes
        self.input_size = input_size
        self.grayscale = grayscale
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Charge le modèle depuis le fichier
        
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            if not TF_AVAILABLE:
                print("TensorFlow n'est pas disponible")
                print("Mode demo simple (predictions aleatoires)")
                self._create_simple_demo_model()
                return True
                
            if not os.path.exists(self.model_path):
                print(f"ATTENTION: Le fichier du modele n'existe pas: {self.model_path}")
                print("Veuillez placer votre modele CNN (.h5) dans le dossier 'model/'")
                print("   Pour l'instant, un modele de demonstration sera utilise.")
                
                # Créer un modèle de démonstration simple
                self._create_demo_model()
                return True
            
            print(f"Chargement du modele depuis: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            self.is_loaded = True
            print(f"Modele charge avec succes!")
            print(f"   Classes: {self.classes}")
            print(f"   Taille d'entrée: {self.input_size}")
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement du modele: {e}")
            print("Creation d'un modele de demonstration...")
            if TF_AVAILABLE:
                self._create_demo_model()
            else:
                self._create_simple_demo_model()
            return True
    
    def _create_demo_model(self):
        """
        Crée un modèle de démonstration simple pour tester l'application
        sans avoir besoin d'un vrai modèle entraîné
        """
        print("Creation d'un modele de demonstration...")
        
        # Créer un modèle CNN simple (non entraîné)
        input_shape = (*self.input_size, 1 if self.grayscale else 3)
        
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.is_loaded = True
        
        self.is_loaded = True
        
        print("Modele de demonstration cree!")
        print("Note: Ce modele n'est PAS entraine. Les predictions seront aleatoires.")
        print("   Pour utiliser un vrai modele, placez votre fichier .h5 dans le dossier 'model/'")
    
    def _create_simple_demo_model(self):
        """
        Crée un modèle de démonstration ultra-simple qui ne nécessite pas TensorFlow
        Génère des prédictions aléatoires
        """
        print("Creation d'un modele de demonstration simple...")
        
        # Pas de vrai modèle, juste un flag
        self.model = "SIMPLE_DEMO"
        self.is_loaded = True
        
        print("Modele de demonstration simple cree!")
        print("Note: TensorFlow n'est pas disponible. Les predictions seront aleatoires.")
        print("   Pour installer TensorFlow: pip install tensorflow")

    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Prétraite une image pour la prédiction
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Tableau numpy prétraité, prêt pour le modèle
        """
        # Charger l'image
        img = Image.open(image_path)
        
        # Convertir en niveaux de gris si nécessaire
        if self.grayscale:
            img = img.convert('L')  # L = Grayscale
        else:
            img = img.convert('RGB')
        
        # Redimensionner à la taille attendue par le modèle
        img = img.resize(self.input_size)
        
        # Convertir en array numpy
        img_array = np.array(img)
        
        # Normaliser les pixels (0-255 -> 0-1)
        img_array = img_array.astype('float32') / 255.0
        
        # Ajouter la dimension du canal si nécessaire
        if self.grayscale:
            img_array = np.expand_dims(img_array, axis=-1)
        
        # Ajouter la dimension du batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Effectue une prédiction sur une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Tuple (predicted_label, confidence, all_probabilities)
            - predicted_label: La classe prédite
            - confidence: Score de confiance (0-1)
            - all_probabilities: Dictionnaire {classe: probabilité}
        """
        if not self.is_loaded:
            raise RuntimeError("Le modèle n'est pas chargé. Appelez load_model() d'abord.")
        
        # Si c'est le modèle simple demo (sans TensorFlow)
        if self.model == "SIMPLE_DEMO":
            # Générer des probabilités aléatoires
            import random
            probabilities = np.random.dirichlet(np.ones(len(self.classes)))
            
            # Trouver la classe avec la probabilité maximale
            predicted_index = np.argmax(probabilities)
            predicted_label = self.classes[predicted_index]
            confidence = float(probabilities[predicted_index])
            
            # Créer le dictionnaire de toutes les probabilités
            all_probabilities = {
                self.classes[i]: float(probabilities[i])
                for i in range(len(self.classes))
            }
            
            return predicted_label, confidence, all_probabilities
        
        # Sinon, utiliser le vrai modèle TensorFlow
        # Prétraiter l'image
        processed_image = self.preprocess_image(image_path)
        
        # Faire la prédiction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Extraire les probabilités (première prédiction du batch)
        probabilities = predictions[0]
        
        # Trouver la classe avec la probabilité maximale
        predicted_index = np.argmax(probabilities)
        predicted_label = self.classes[predicted_index]
        confidence = float(probabilities[predicted_index])
        
        # Créer le dictionnaire de toutes les probabilités
        all_probabilities = {
            self.classes[i]: float(probabilities[i])
            for i in range(len(self.classes))
        }
        
        return predicted_label, confidence, all_probabilities
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Retourne les informations sur le modèle
        
        Returns:
            Dictionnaire avec les informations du modèle
        """
        if not self.is_loaded:
            return {"error": "Modèle non chargé"}
        
        return {
            "classes": self.classes,
            "num_classes": len(self.classes),
            "input_size": self.input_size,
            "grayscale": self.grayscale,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded
        }
