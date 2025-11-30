"""
Configuration de l'application Flask et MongoDB
"""
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env (si présent)
load_dotenv()

# Configuration Flask
class Config:
    """Configuration principale de l'application"""
    
    # Clé secrète pour Flask (sessions, CSRF, etc.)
    SECRET_KEY = os.getenv('SECRET_KEY', 'votre_cle_secrete_a_changer_en_production')
    
    # Taille maximale des fichiers uploadés (16 MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    # Dossier de stockage des images uploadées
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    
    # Extensions de fichiers autorisées
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Configuration MongoDB
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'cnn_interface')
    COLLECTION_PREDICTIONS = 'predictions'
    COLLECTION_USERS = 'users'
    
    # Configuration du modèle CNN
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'cnn_model.h5')
    
    # Classes du modèle - Sign Language CNN (36 classes: 0-9 et 26 lettres)
    # Le modèle a été entraîné pour reconnaître les chiffres et lettres du langage des signes
    default_classes = '0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'
    MODEL_CLASSES = os.getenv('MODEL_CLASSES', default_classes).split(',')
    
    # Taille d'entrée attendue par le modèle (largeur, hauteur)
    # Le modèle Sign Language CNN attend des images 64x64 pixels
    MODEL_INPUT_SIZE = (64, 64)
    
    # Le modèle attend-il des images en couleur (3 canaux) ou en niveaux de gris (1 canal) ?
    # Le modèle Sign Language CNN utilise des images RGB (couleur)
    MODEL_GRAYSCALE = False


def allowed_file(filename):
    """
    Vérifie si l'extension du fichier est autorisée
    
    Args:
        filename (str): Nom du fichier
        
    Returns:
        bool: True si l'extension est autorisée
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
