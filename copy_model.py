"""
Script pour vérifier et copier le modèle CNN depuis le notebook vers le projet
"""
import os
import shutil
from pathlib import Path

# Chemins
NOTEBOOK_DIR = Path(r"C:\Users\yassi\Downloads")
PROJECT_MODEL_DIR = Path(r"C:\Users\yassi\.gemini\antigravity\scratch\cnn-mongodb-project\model")

# Noms de fichiers possibles
MODEL_FILENAMES = [
    "sign_language_cnn.h5",
    "sign_language_cnn.keras",
    "model.h5"
]

TARGET_NAME = "cnn_model.h5"

def find_model_file():
    """Cherche le fichier du modèle dans Downloads"""
    for filename in MODEL_FILENAMES:
        filepath = NOTEBOOK_DIR / filename
        if filepath.exists():
            print(f"✅ Modèle trouvé: {filepath}")
            return filepath
    return None

def copy_model(source_path):
    """Copie le modèle vers le dossier du projet"""
    target_path = PROJECT_MODEL_DIR / TARGET_NAME
    
    # Créer le dossier si nécessaire
    PROJECT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copier le fichier
    print(f"📋 Copie de {source_path.name} vers {target_path}")
    shutil.copy2(source_path, target_path)
    print(f"✅ Modèle copié avec succès!")
    print(f"📁 Emplacement: {target_path}")
    return target_path

def verify_model():
    """Vérifie que le modèle existe dans le projet"""
    target_path = PROJECT_MODEL_DIR / TARGET_NAME
    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"✅ Le modèle est déjà présent dans le projet")
        print(f"📁 Emplacement: {target_path}")
        print(f"📊 Taille: {size_mb:.2f} MB")
        return True
    return False

def main():
    print("=" * 60)
    print("🔍 Vérification du modèle CNN Sign Language")
    print("=" * 60)
    print()
    
    # Vérifier si le modèle existe déjà dans le projet
    if verify_model():
        print("\n✅ Le modèle est déjà configuré!")
        print("Vous pouvez lancer l'application avec: python app.py")
        return
    
    print("⚠️  Le modèle n'est pas encore dans le projet")
    print(f"🔍 Recherche dans: {NOTEBOOK_DIR}")
    print()
    
    # Chercher le modèle
    model_path = find_model_file()
    
    if model_path:
        print()
        response = input("❓ Voulez-vous copier ce modèle vers le projet? (O/n): ")
        if response.lower() in ['o', 'oui', 'y', 'yes', '']:
            copy_model(model_path)
            print()
            print("=" * 60)
            print("✅ Configuration terminée!")
            print("=" * 60)
            print()
            print("Prochaines étapes:")
            print("1. Vérifier que MongoDB est démarré")
            print("2. Lancer l'application: python app.py")
            print("3. Ouvrir http://localhost:5000")
        else:
            print("❌ Opération annulée")
    else:
        print()
        print("❌ Modèle non trouvé dans le dossier Downloads")
        print()
        print("📝 Instructions:")
        print("1. Téléchargez sign_language_cnn.h5 depuis Colab/Kaggle")
        print("2. Placez-le dans C:\\Users\\yassi\\Downloads\\")
        print("3. Relancez ce script")
        print()
        print("Ou copiez manuellement le fichier vers:")
        print(f"   {PROJECT_MODEL_DIR / TARGET_NAME}")

if __name__ == "__main__":
    main()
