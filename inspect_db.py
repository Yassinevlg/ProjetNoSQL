"""
Script simple pour inspecter le contenu de la base de données MongoDB
"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pprint import pprint

# Charger la config
load_dotenv()
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DATABASE_NAME', 'cnn_interface')

def inspect_db():
    print(f"Connexion a {MONGO_URI}...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info() # Test connexion
        print("Connexion reussie!")
        
        db = client[DB_NAME]
        print(f"\nBase de donnees: {DB_NAME}")
        
        # Lister les collections
        collections = db.list_collection_names()
        print(f"Collections trouvees: {collections}")
        
        for col_name in collections:
            col = db[col_name]
            count = col.count_documents({})
            print(f"\n   Collection '{col_name}' ({count} documents):")
            
            # Afficher les 3 derniers documents
            recent_docs = list(col.find().sort('_id', -1).limit(3))
            if recent_docs:
                print("      Derniers ajouts:")
                for doc in recent_docs:
                    # Simplifier l'affichage pour la lisibilité
                    display_doc = {k: v for k, v in doc.items() if k != 'all_probabilities'}
                    print(f"      - {display_doc}")
            else:
                print("      (Vide)")
                
    except Exception as e:
        print(f"Erreur: {e}")
        print("Assurez-vous que MongoDB est bien lance !")

if __name__ == "__main__":
    inspect_db()
