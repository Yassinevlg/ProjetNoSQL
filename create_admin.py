"""
Script pour créer un administrateur ou promouvoir un utilisateur existant
"""
from utils.db_manager import DatabaseManager
from config import Config
from werkzeug.security import generate_password_hash

def create_admin():
    db_manager = DatabaseManager(Config.MONGO_URI, Config.DATABASE_NAME)
    
    print("--- Gestion Administrateur ---")
    choice = input("1. Créer un nouvel admin\n2. Promouvoir un utilisateur existant\nChoix: ")
    
    if choice == '1':
        username = input("Username: ")
        email = input("Email: ")
        password = input("Password: ")
        
        # Vérifier si existe
        if db_manager.get_user_by_email(email):
            print("Cet email existe déjà.")
            return

        user_id = db_manager.create_user(username, email, password, role='admin')
        if user_id:
            print(f"✅ Admin créé avec succès ! ID: {user_id}")
        else:
            print("❌ Erreur lors de la création.")
            
    elif choice == '2':
        email = input("Email de l'utilisateur à promouvoir: ")
        user = db_manager.get_user_by_email(email)
        
        if user:
            db_manager.users.update_one(
                {"email": email},
                {"$set": {"role": "admin"}}
            )
            print(f"✅ L'utilisateur {user['username']} est maintenant ADMIN.")
        else:
            print("❌ Utilisateur non trouvé.")
            
    else:
        print("Choix invalide.")

if __name__ == "__main__":
    create_admin()
