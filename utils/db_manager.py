"""
Gestionnaire de base de données MongoDB
Gère toutes les opérations CRUD et agrégations pour le projet CNN
"""
from pymongo import MongoClient, DESCENDING
from datetime import datetime
from bson.objectid import ObjectId
from typing import Dict, List, Optional, Any
from werkzeug.security import generate_password_hash, check_password_hash


class DatabaseManager:
    """Gestionnaire de connexion et opérations MongoDB"""
    
    def __init__(self, mongo_uri: str, database_name: str):
        """
        Initialise la connexion à MongoDB
        
        Args:
            mongo_uri: URI de connexion MongoDB
            database_name: Nom de la base de données
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Vérifier la connexion
            self.client.server_info()
            
            self.db = self.client[database_name]
            self.predictions = self.db['predictions']
            self.users = self.db['users']
            self.training_runs = self.db['training_runs']
            print(f"Connecte a MongoDB: {database_name}")
            self.is_connected = True
            
        except Exception as e:
            print(f"Erreur de connexion MongoDB: {e}")
            # On ne lève pas d'erreur pour permettre à l'app de démarrer
            self.client = None
            self.db = None
            self.predictions = None
            self.users = None
            self.training_runs = None
            self.is_connected = False

    def close(self):
        """Ferme la connexion MongoDB"""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()

    # ========== GESTION DES UTILISATEURS ==========

    def create_user(self, username, email, password, role='user'):
        """Crée un nouvel utilisateur avec mot de passe haché"""
        if not self.is_connected:
            return None
            
        # Vérifier si l'utilisateur existe déjà
        if self.users.find_one({"$or": [{"email": email}, {"username": username}]}):
            return False  # Existe déjà
            
        user_doc = {
            "username": username,
            "email": email,
            "password_hash": generate_password_hash(password),
            "role": role,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
        result = self.users.insert_one(user_doc)
        return str(result.inserted_id)

    def get_user_by_email(self, email):
        """Récupère un utilisateur par email"""
        if not self.is_connected:
            return None
        return self.users.find_one({"email": email})
        
    def get_user_by_id(self, user_id):
        """Récupère un utilisateur par ID"""
        if not self.is_connected:
            return None
        try:
            return self.users.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None

    def check_password(self, user_doc, password):
        """Vérifie le mot de passe d'un utilisateur"""
        if not user_doc or 'password_hash' not in user_doc:
            return False
        return check_password_hash(user_doc['password_hash'], password)
        
    def get_all_users(self):
        """Récupère tous les utilisateurs (pour admin)"""
        if not self.is_connected:
            return []
        # On exclut le hash du mot de passe
        return list(self.users.find({}, {"password_hash": 0}))

    def get_predictions_by_user(self, user_id):
        """Récupère les prédictions d'un utilisateur spécifique"""
        if not self.is_connected:
            return []
        try:
            return list(
                self.predictions
                .find({"user_id": ObjectId(user_id)})
                .sort("created_at", DESCENDING)
            )
        except Exception:
            return []

    # ========== OPÉRATIONS D'INSERTION ==========
    
    def insert_prediction(
        self,
        image_path: str,
        original_filename: str,
        predicted_label: str,
        confidence: float,
        all_probabilities: Dict[str, float],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """
        Insère une nouvelle prédiction dans la base de données
        """
        prediction_doc = {
            "image_path": image_path,
            "original_filename": original_filename,
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "all_probabilities": all_probabilities,
            "created_at": datetime.utcnow(),
            "user_feedback": {
                "is_correct": None,
                "true_label": None,
                "feedback_at": None
            },
            "meta": {
                "ip": ip_address,
                "user_agent": user_agent
            }
        }
        
        # Ajouter l'ID utilisateur si connecté
        if user_id:
            try:
                prediction_doc["user_id"] = ObjectId(user_id)
            except Exception:
                # On ignore si l'ID est invalide
                pass
        
        if self.is_connected and self.predictions is not None:
            try:
                result = self.predictions.insert_one(prediction_doc)
                return str(result.inserted_id)
            except Exception as e:
                print(f"Erreur insert MongoDB: {e}")
                return "error_id"
        return "mock_id"
    
    # ========== OPÉRATIONS DE LECTURE ==========
    
    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Récupère une prédiction par son ID"""
        if not self.is_connected:
            return None
        try:
            return self.predictions.find_one({"_id": ObjectId(prediction_id)})
        except Exception:
            return None
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les dernières prédictions (triées par date décroissante)"""
        if not self.is_connected:
            return []
        predictions = (
            self.predictions
            .find()
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        return list(predictions)
    
    def get_predictions_by_label(self, predicted_label: str) -> List[Dict[str, Any]]:
        """Récupère toutes les prédictions pour une classe donnée"""
        if not self.is_connected:
            return []
        predictions = self.predictions.find({"predicted_label": predicted_label})
        return list(predictions)
    
    def get_predictions_with_feedback(self) -> List[Dict[str, Any]]:
        """Récupère uniquement les prédictions pour lesquelles un feedback a été fourni"""
        if not self.is_connected:
            return []
        predictions = self.predictions.find({
            "user_feedback.is_correct": {"$ne": None}
        })
        return list(predictions)
    
    # ========== OPÉRATIONS DE MISE À JOUR ==========
    
    def update_feedback(
        self,
        prediction_id: str,
        is_correct: bool,
        true_label: Optional[str] = None
    ) -> bool:
        """
        Met à jour le feedback utilisateur pour une prédiction
        """
        if not self.is_connected:
            return False

        try:
            update_data = {
                "$set": {
                    "user_feedback.is_correct": is_correct,
                    "user_feedback.true_label": true_label,
                    "user_feedback.feedback_at": datetime.utcnow()
                }
            }

            result = self.predictions.update_one(
                {"_id": ObjectId(prediction_id)},
                update_data
            )

            return result.modified_count > 0
        except Exception as e:
            print(f"Erreur lors de la mise à jour du feedback: {e}")
            return False

    # ========== GESTIONS DES UTILISATEURS (MISES A JOUR) ==========

    def update_user_role(self, user_id: str, new_role: str) -> bool:
        """Modifie le rôle d'un utilisateur (ex: 'user' <-> 'admin')"""
        if not self.is_connected:
            return False

        try:
            result = self.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"role": new_role}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Erreur update_user_role: {e}")
            return False

    def set_user_active(self, user_id: str, is_active: bool) -> bool:
        """Active ou désactive un compte utilisateur"""
        if not self.is_connected:
            return False

        try:
            result = self.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"is_active": bool(is_active)}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Erreur set_user_active: {e}")
            return False

    def set_last_login(self, user_id: str) -> bool:
        """Met à jour le champ `last_login` pour un utilisateur donné"""
        if not self.is_connected:
            return False

        try:
            result = self.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Erreur set_last_login: {e}")
            return False

    def get_predictions(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Récupère des prédictions selon un filtre MongoDB (utilisé par l'admin)
        """
        if not self.is_connected:
            return []

        try:
            if filter_query is None:
                filter_query = {}
            cursor = (
                self.predictions
                .find(filter_query)
                .sort("created_at", DESCENDING)
                .limit(limit)
            )
            return list(cursor)
        except Exception as e:
            print(f"Erreur get_predictions: {e}")
            return []
    
    # ========== OPÉRATIONS DE SUPPRESSION ==========
    
    def delete_prediction(self, prediction_id: str) -> bool:
        """Supprime une prédiction par son ID"""
        if not self.is_connected:
            return False
            
        try:
            result = self.predictions.delete_one({"_id": ObjectId(prediction_id)})
            return result.deleted_count > 0
        except Exception:
            return False
    
    def delete_all_predictions(self) -> int:
        """Supprime toutes les prédictions (utile pour les tests)"""
        if not self.is_connected:
            return 0
        result = self.predictions.delete_many({})
        return result.deleted_count
    
    # ========== AGRÉGATIONS ET STATISTIQUES ==========
    
    def get_total_predictions(self) -> int:
        """Retourne le nombre total de prédictions"""
        if not self.is_connected:
            return 0
        return self.predictions.count_documents({})
    
    def get_predictions_count_by_label(self) -> List[Dict[str, Any]]:
        """
        Nombre de prédictions par classe prédite (avec confiance moyenne)
        """
        if not self.is_connected:
            return []
            
        pipeline = [
            {
                "$group": {
                    "_id": "$predicted_label",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"}
                }
            },
            {"$sort": {"count": -1}}
        ]
        
        return list(self.predictions.aggregate(pipeline))
    
    def get_true_labels_distribution(self) -> List[Dict[str, Any]]:
        """
        Distribution des vraies classes (selon les feedbacks utilisateur)
        """
        if not self.is_connected:
            return []
            
        pipeline = [
            {
                "$match": {
                    "user_feedback.true_label": {"$ne": None}
                }
            },
            {
                "$group": {
                    "_id": "$user_feedback.true_label",
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"count": -1}}
        ]
        
        return list(self.predictions.aggregate(pipeline))
    
    def get_accuracy_rate(self) -> Optional[Dict[str, Any]]:
        """
        Calcule le taux de bonnes prédictions (basé sur les feedbacks)
        
        Returns:
            Dictionnaire avec total, correct, incorrect, accuracy (en %)
            Retourne None s'il n'y a pas de feedback
        """
        if not self.is_connected:
            return None
            
        pipeline = [
            {
                "$match": {
                    "user_feedback.is_correct": {"$ne": None}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "correct": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$user_feedback.is_correct", True]},
                                1,
                                0
                            ]
                        }
                    },
                    "incorrect": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$user_feedback.is_correct", False]},
                                1,
                                0
                            ]
                        }
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "total": 1,
                    "correct": 1,
                    "incorrect": 1,
                    "accuracy": {
                        "$cond": [
                            {"$eq": ["$total", 0]},
                            0,
                            {
                                "$multiply": [
                                    {"$divide": ["$correct", "$total"]},
                                    100
                                ]
                            }
                        ]
                    }
                }
            }
        ]
        
        results = list(self.predictions.aggregate(pipeline))
        return results[0] if results else None
    
    def get_confusion_data(self) -> List[Dict[str, Any]]:
        """
        Récupère les données pour construire une matrice de confusion
        (prédictions vs vraies classes selon feedback)
        """
        if not self.is_connected:
            return []
            
        pipeline = [
            {
                "$match": {
                    "user_feedback.true_label": {"$ne": None}
                }
            },
            {
                "$group": {
                    "_id": {
                        "predicted": "$predicted_label",
                        "true": "$user_feedback.true_label"
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "predicted": "$_id.predicted",
                    "true": "$_id.true",
                    "count": 1
                }
            }
        ]
        
        return list(self.predictions.aggregate(pipeline))

    def get_per_class_accuracy(self) -> List[Dict[str, Any]]:
        """
        Accuracy par classe (basé sur les prédictions annotées avec feedback).
        Retourne une liste de dicts :
        { label, total, correct, accuracy (0-1) }
        """
        if not self.is_connected:
            return []

        pipeline = [
            {
                "$match": {
                    "user_feedback.is_correct": {"$ne": None}
                }
            },
            {
                "$group": {
                    "_id": "$predicted_label",
                    "total": {"$sum": 1},
                    "correct": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$user_feedback.is_correct", True]},
                                1,
                                0
                            ]
                        }
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "label": "$_id",
                    "total": 1,
                    "correct": 1,
                    "accuracy": {
                        "$cond": [
                            {"$eq": ["$total", 0]},
                            0,
                            {"$divide": ["$correct", "$total"]}
                        ]
                    }
                }
            },
            {"$sort": {"label": 1}}
        ]

        return list(self.predictions.aggregate(pipeline))
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques globales pour le dashboard
        """
        total = self.get_total_predictions()
        accuracy_data = self.get_accuracy_rate()
        predictions_by_label = self.get_predictions_count_by_label()
        per_class_accuracy = self.get_per_class_accuracy()
        true_labels_dist = self.get_true_labels_distribution()
        
        return {
            "total_predictions": total,
            "accuracy": accuracy_data,            # global (en %)
            "predictions_by_label": predictions_by_label,
            "per_class_accuracy": per_class_accuracy,
            "true_labels_distribution": true_labels_dist,
            "has_feedback": accuracy_data is not None
        }

    # ========== GESTION DES ENTRAÎNEMENTS (TRAINING RUNS) ==========

    def get_feedback_data_for_training(self) -> List[Dict[str, Any]]:
        """
        Récupère toutes les prédictions qui ont un feedback utilisateur complet (true_label).
        Ces données seront utilisées pour ré-entraîner le modèle.
        """
        if not self.is_connected:
            return []
        
        # On cherche les documents où user_feedback.true_label existe et n'est pas null
        query = {
            "user_feedback.true_label": {"$ne": None}
        }
        
        # On ne récupère que les champs nécessaires
        projection = {
            "image_path": 1,
            "user_feedback.true_label": 1,
            "_id": 0
        }
        
        return list(self.predictions.find(query, projection))

    def create_training_run(self, base_dataset_info: str = "MNIST + Feedback") -> str:
        """
        Crée un nouveau log de session d'entraînement.
        Statut initial: 'running'
        """
        if not self.is_connected:
            return "mock_run_id"
            
        run_doc = {
            "started_at": datetime.utcnow(),
            "ended_at": None,
            "status": "running",
            "base_dataset_info": base_dataset_info,
            "used_feedback_count": 0,
            "model_path": None,
            "error_message": None
        }
        
        result = self.training_runs.insert_one(run_doc)
        return str(result.inserted_id)

    def update_training_run(
        self, 
        run_id: str, 
        status: str, 
        model_path: Optional[str] = None, 
        used_feedback_count: int = 0,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Met à jour un log d'entraînement (généralement à la fin).
        """
        if not self.is_connected:
            return False
            
        update_data = {
            "$set": {
                "status": status,
                "ended_at": datetime.utcnow(),
                "used_feedback_count": used_feedback_count
            }
        }
        
        if model_path:
            update_data["$set"]["model_path"] = model_path
            
        if error_message:
            update_data["$set"]["error_message"] = error_message
            
        try:
            result = self.training_runs.update_one(
                {"_id": ObjectId(run_id)},
                update_data
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Erreur update_training_run: {e}")
            return False

    def update_training_progress(
        self,
        run_id: str,
        progress: float,
        current_epoch: int,
        total_epochs: int,
        message: str = ""
    ) -> bool:
        """
        Met à jour la progression d'un entraînement en cours.
        """
        if not self.is_connected:
            return False
            
        update_data = {
            "$set": {
                "progress": progress,
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "message": message
            }
        }
        
        try:
            result = self.training_runs.update_one(
                {"_id": ObjectId(run_id)},
                update_data
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Erreur update_training_progress: {e}")
            return False

    def get_all_training_runs(self) -> List[Dict[str, Any]]:
        """Récupère l'historique des entraînements"""
        if not self.is_connected:
            return []
        return list(self.training_runs.find().sort("started_at", DESCENDING))
