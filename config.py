"""
Configuration de l'application Flask et MongoDB
"""
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env (si présent)
load_dotenv()

# Répertoire de base du projet (là où se trouve config.py)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Configuration principale de l'application"""
    
    # Clé secrète pour Flask (sessions, CSRF, etc.)
    SECRET_KEY = os.getenv('SECRET_KEY', 'votre_cle_secrete_a_changer_en_production')
    
    # Taille maximale des fichiers uploadés (16 MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    # Dossier de stockage des images uploadées (dans static/uploads)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    
    # Extensions de fichiers autorisées
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Configuration MongoDB
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'cnn_interface')
    COLLECTION_PREDICTIONS = 'predictions'
    COLLECTION_USERS = 'users'
    
    # === Configuration du modèle CNN ===
    # Chemin vers ton vrai fichier de modèle .h5
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'sign_language_cnn.h5')
    MODEL_PATH = os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH)
    
    # Classes du modèle - ordre EXACT utilisé à l'entraînement
    # (tu peux adapter cette liste si nécessaire, ou la surcharger via la variable d'environnement MODEL_CLASSES)
        # Classes du modèle - ordre EXACT utilisé à l'entraînement
    # Ici : chiffres 1-9 + lettres A-Z (sans 0)
    default_classes = (
        '1,2,3,4,5,6,7,8,9,'
        'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,'
        'P,Q,R,S,T,U,V,W,X,Y,Z'
    )
    MODEL_CLASSES = os.getenv('MODEL_CLASSES', default_classes).split(',')
    
    # Taille d'entrée attendue par le modèle (hauteur, largeur)
    # Ton modèle Keras attend des images de forme (64, 64, 3)
    MODEL_INPUT_SIZE = (64, 64)
    
    # Le modèle utilise des images RGB (3 canaux) → False = pas de niveaux de gris
    MODEL_GRAYSCALE = False


def allowed_file(filename: str) -> bool:
    """
    Vérifie si l'extension du fichier est autorisée
    
    Args:
        filename (str): Nom du fichier
        
    Returns:
        bool: True si l'extension est autorisée
    """
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )
