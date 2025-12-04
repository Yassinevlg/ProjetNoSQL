import os
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify
)
from werkzeug.utils import secure_filename

from config import Config, allowed_file
from utils.db_manager import DatabaseManager
from model.model_loader import CNNModelLoader
from model.trainer import ModelTrainer

# Initialisation de l'application Flask
app = Flask(__name__)
app.config.from_object(Config)

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialiser le gestionnaire de base de données
db_manager = DatabaseManager(
    mongo_uri=app.config['MONGO_URI'],
    database_name=app.config['DATABASE_NAME']
)

# Initialiser le trainer
model_trainer = ModelTrainer(
    db_manager=db_manager,
    model_dir=os.path.join(os.path.dirname(__file__), 'model'),
    classes=app.config['MODEL_CLASSES']
)

# Initialiser et charger le modèle CNN
cnn_model = CNNModelLoader(
    model_path=app.config['MODEL_PATH'],
    classes=app.config['MODEL_CLASSES'],
    input_size=app.config['MODEL_INPUT_SIZE'],
    grayscale=app.config['MODEL_GRAYSCALE']
)

# Charger le modèle au démarrage
print("=" * 60)
print("Demarrage de l'application CNN + MongoDB")
print("=" * 60)
cnn_model.load_model()
print("=" * 60)

print("MODEL_PATH utilisé :", app.config['MODEL_PATH'])
print("Classes du modèle (depuis app.config) :", app.config['MODEL_CLASSES'])

# Essayer d'afficher l'état du mode démo si l'attribut existe
print("Attributs du CNNModelLoader :", dir(cnn_model))
print("Mode démo ? (demo_mode) :", getattr(cnn_model, "demo_mode", "inconnu"))
print("Mode démo ? (is_demo)   :", getattr(cnn_model, "is_demo", "inconnu"))


# ========== DÉCORATEURS ET UTILITAIRES ==========

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Veuillez vous connecter pour accéder à cette page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'admin':
            flash('Accès réservé aux administrateurs.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_user():
    """Injecte l'utilisateur courant dans tous les templates"""
    user = None
    if 'user_id' in session:
        user = db_manager.get_user_by_id(session['user_id'])
    return dict(current_user=user)


# ========== ROUTES D'AUTHENTIFICATION ==========

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Créer l'utilisateur (par défaut 'user')
        user_id = db_manager.create_user(username, email, password)
        
        if user_id:
            flash('Compte créé avec succès ! Veuillez vous connecter.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Erreur: Cet email ou nom d\'utilisateur existe déjà.', 'error')
            
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = db_manager.get_user_by_email(email)
        
        if user and db_manager.check_password(user, password):
            session['user_id'] = str(user['_id'])
            session['role'] = user['role']
            session['username'] = user['username']
            # Mettre à jour la date de dernière connexion
            try:
                db_manager.set_last_login(str(user['_id']))
            except Exception:
                pass

            flash(f'Bienvenue {user["username"]} !', 'success')
            return redirect(url_for('index'))
        else:
            flash('Email ou mot de passe incorrect.', 'error')
            
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('login'))


# ========== ROUTES DE L'APPLICATION ==========

@app.route('/')
def index():
    """Page d'accueil - redirige vers login ou predict selon l'état de connexion"""
    if 'user_id' in session:
        # Utilisateur connecté : rediriger vers la page de prédiction
        return redirect(url_for('predict'))
    else:
        # Utilisateur non connecté : rediriger vers la page de connexion
        return redirect(url_for('login'))


@app.route('/project')
def project():
    """Page de description du projet avec architecture CNN et graphiques d'entraînement"""
    model_info = cnn_model.get_model_info()
    
    # Récupérer l'historique d'entraînement du dernier training run réussi
    training_runs = db_manager.get_all_training_runs()
    training_history = {
        'epochs': [1, 2, 3, 4, 5],
        'accuracy': [0.85, 0.92, 0.95, 0.97, 0.98],
        'val_accuracy': [0.82, 0.88, 0.91, 0.93, 0.95],
        'loss': [0.45, 0.25, 0.15, 0.10, 0.07],
        'val_loss': [0.55, 0.35, 0.25, 0.20, 0.15]
    }
    
    # Chercher l'historique réel dans le dernier run réussi
    for run in training_runs:
        if run.get('status') == 'success' and run.get('training_history'):
            history = run['training_history']
            training_history = {
                'epochs': list(range(1, len(history.get('accuracy', [])) + 1)),
                'accuracy': history.get('accuracy', []),
                'val_accuracy': history.get('val_accuracy', []),
                'loss': history.get('loss', []),
                'val_loss': history.get('val_loss', [])
            }
            break
    
    return render_template('project.html', 
                           model_info=model_info, 
                           training_history=training_history)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Page de prédiction - Upload d'image et classification"""
    
    if request.method == 'GET':
        # Afficher le formulaire d'upload
        model_info = cnn_model.get_model_info()
        return render_template('predict.html', model_info=model_info)
    
    # Traitement POST - Upload et prédiction
    if 'image' not in request.files:
        flash('Aucun fichier sélectionné', 'error')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        flash('Aucun fichier sélectionné', 'error')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash(f'Type de fichier non autorisé. Extensions autorisées: {", ".join(Config.ALLOWED_EXTENSIONS)}', 'error')
        return redirect(request.url)
    
    try:
        # Sauvegarder l'image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Faire la prédiction
        predicted_label, confidence, all_probabilities = cnn_model.predict(filepath)
        
        # Sauvegarder dans MongoDB
        prediction_id = db_manager.insert_prediction(
            image_path=filepath,
            original_filename=filename,
            predicted_label=predicted_label,
            confidence=confidence,
            all_probabilities=all_probabilities,
            user_id=session.get('user_id'),  # Lier à l'utilisateur
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string
        )
        
        # Rediriger vers la page de résultat
        return redirect(url_for('result', prediction_id=prediction_id))
        
    except Exception as e:
        flash(f'Erreur lors de la prédiction: {str(e)}', 'error')
        print(f"Erreur de prédiction: {e}")
        return redirect(url_for('predict'))


@app.route('/result/<prediction_id>')
def result(prediction_id):
    """Page affichant le résultat d'une prédiction"""
    try:
        prediction = db_manager.get_prediction_by_id(prediction_id)
        
        if not prediction:
            flash('Prédiction non trouvée', 'error')
            return redirect(url_for('index'))
        
        # Convertir le chemin absolu en chemin relatif pour l'affichage
        image_url = prediction['image_path'].replace('\\', '/').split('/static/')[-1]
        image_url = f"/static/{image_url}"
        
        return render_template(
            'result.html',
            prediction=prediction,
            image_url=image_url,
            prediction_id=str(prediction['_id'])
        )
        
    except Exception as e:
        flash(f'Erreur lors de la récupération du résultat: {str(e)}', 'error')
        print(f"Erreur result: {e}")
        return redirect(url_for('index'))


@app.route('/history')
@login_required
def history():
    """Page affichant l'historique des prédictions de l'utilisateur"""
    try:
        # Récupérer les prédictions de l'utilisateur connecté
        user_id = session['user_id']
        predictions = db_manager.get_predictions_by_user(user_id)
        
        # Convertir les chemins d'images pour l'affichage
        for pred in predictions:
            pred['image_url'] = pred['image_path'].replace('\\', '/').split('/static/')[-1]
            pred['image_url'] = f"/static/{pred['image_url']}"
        
        return render_template('history.html', predictions=predictions)
        
    except Exception as e:
        flash(f'Erreur lors de la récupération de l\'historique: {str(e)}', 'error')
        print(f"Erreur history: {e}")
        return render_template('history.html', predictions=[])





# ========== GESTIONNAIRES D'ERREURS ==========

@app.errorhandler(404)
def page_not_found(e):
    """Page 404"""
    return render_template('error.html', error_code=404, error_message="Page non trouvée"), 404


@app.errorhandler(500)
def internal_error(e):
    """Page 500"""
    return render_template('error.html', error_code=500, error_message="Erreur interne du serveur"), 500


# ========== FILTRES JINJA PERSONNALISÉS ==========

@app.template_filter('datetime')
def format_datetime(value):
    """Formater un datetime pour l'affichage"""
    if value is None:
        return ""
    return value.strftime('%d/%m/%Y %H:%M:%S')


@app.template_filter('percentage')
def format_percentage(value):
    """Formater un nombre décimal en pourcentage"""
    if value is None:
        return "0%"
    return f"{value * 100:.2f}%"




@app.route('/feedback/<prediction_id>', methods=['POST'])
def feedback(prediction_id):
    """Endpoint pour soumettre un feedback sur une prédiction"""
    try:
        is_correct = request.form.get('is_correct') == 'true'
        true_label = request.form.get('true_label', None)
        
        # Mettre à jour le feedback
        success = db_manager.update_feedback(
            prediction_id=prediction_id,
            is_correct=is_correct,
            true_label=true_label if not is_correct else None
        )
        
        if success:
            flash('Merci pour votre feedback!', 'success')
        else:
            flash('Erreur lors de l\'enregistrement du feedback', 'error')
        # Rediriger vers la page du résultat
        return redirect(request.referrer or url_for('result', prediction_id=prediction_id))
    except Exception as e:
        print(f"Erreur feedback: {e}")
        flash('Erreur lors du traitement du feedback', 'error')
        return redirect(request.referrer or url_for('history'))


@app.route('/my_predictions')
@login_required
def my_predictions():
    """Alias vers l'historique personnel (compatibilité user story)."""
    return redirect(url_for('history'))


def api_stats():
    """API endpoint pour obtenir les statistiques en JSON"""
    try:
        stats = db_manager.get_global_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions/recent')
def api_recent_predictions():
    """API endpoint pour obtenir les récentes prédictions en JSON"""
    try:
        limit = request.args.get('limit', 10, type=int)
        predictions = db_manager.get_recent_predictions(limit=limit)
        
        # Convertir ObjectId en string pour JSON
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== ROUTES ADMIN ==========

@app.route('/admin')
@admin_required
def admin():
    """Dashboard admin: liste des utilisateurs et statistiques"""
    try:
        # Récupérer les stats globales
        stats = db_manager.get_global_statistics()
        
        # Récupérer l'historique des entraînements
        training_runs = db_manager.get_all_training_runs()
        print(f"DEBUG: training_runs count: {len(training_runs)}")
        for run in training_runs:
            print(f"DEBUG: run: {run}")
        
        # Récupérer les utilisateurs
        users = db_manager.get_all_users()
        
        # Calculer le nombre d'utilisateurs actifs
        active_users_count = sum(1 for u in users if u.get('is_active'))
        
        # Récupérer le nombre total de prédictions
        total_predictions = stats.get('total_predictions', 0)

        return render_template(
            'admin.html',
            users=users,
            total_predictions=total_predictions,
            active_users_count=active_users_count,
            stats=stats,
            training_runs=training_runs
        )
    except Exception as e:
        print(f"Erreur admin dashboard: {e}")
        flash('Erreur lors du chargement du dashboard admin', 'error')
        return redirect(url_for('index'))


@app.route('/admin/users/<user_id>/role', methods=['POST'])
@admin_required
def admin_change_user_role(user_id):
    """Permet à l'admin de modifier le rôle d'un utilisateur"""
    new_role = request.form.get('role')
    if new_role not in ('user', 'admin'):
        flash('Rôle invalide.', 'error')
        return redirect(url_for('admin'))

    success = db_manager.update_user_role(user_id, new_role)
    if success:
        flash('Rôle mis à jour avec succès.', 'success')
    else:
        flash('Échec de la mise à jour du rôle.', 'error')
    return redirect(url_for('admin'))


@app.route('/admin/predictions')
@admin_required
def admin_predictions():
    """Vue admin: liste globale des prédictions, filtres simples via querystring"""
    try:
        # Filtres possibles: user (username or email), predicted_label, is_correct
        filters = {}
        user_q = request.args.get('user')
        if user_q:
            # Chercher l'utilisateur par username ou email
            user_doc = db_manager.users.find_one({"$or": [{"username": user_q}, {"email": user_q}]})
            if user_doc:
                filters['user_id'] = user_doc['_id']

        predicted_label = request.args.get('predicted_label')
        if predicted_label:
            filters['predicted_label'] = predicted_label

        is_correct = request.args.get('is_correct')
        if is_correct in ('true', 'false'):
            filters['user_feedback.is_correct'] = True if is_correct == 'true' else False

        limit = request.args.get('limit', 200, type=int)
        predictions = db_manager.get_predictions(filter_query=filters, limit=limit)

        # Préparer les URLs d'images
        for pred in predictions:
            try:
                pred['image_url'] = pred['image_path'].replace('\\', '/').split('/static/')[-1]
                pred['image_url'] = f"/static/{pred['image_url']}"
            except Exception:
                pred['image_url'] = ''

        return render_template('admin_predictions.html', predictions=predictions)
    except Exception as e:
        print(f"Erreur admin_predictions: {e}")
        flash('Erreur lors de la récupération des prédictions', 'error')
        return redirect(url_for('admin'))





@app.route('/admin/retrain', methods=['POST'])
@login_required
@admin_required
def admin_retrain():
    """Déclenche le ré-entraînement du modèle"""
    try:
        # Démarrer l'entraînement dans un thread séparé
        import threading
        
        def train_async():
            result = model_trainer.retrain()
            if result['status'] == 'success' and 'model_path' in result:
                try:
                    print(f"🔄 Rechargement du nouveau modèle: {result['model_path']}")
                    cnn_model.model_path = result['model_path']
                    cnn_model.load_model()
                except Exception as e:
                    print(f"⚠️ Erreur lors du rechargement du modèle: {e}")
        
        thread = threading.Thread(target=train_async)
        thread.daemon = True
        thread.start()
        
        # Rediriger vers la page de progression
        flash('Entraînement démarré ! Suivez la progression ci-dessous.', 'info')
        return redirect(url_for('admin_training_progress'))
            
    except Exception as e:
        flash(f"Erreur inattendue: {str(e)}", 'error')
        return redirect(url_for('admin'))


@app.route('/admin/training_progress')
@login_required
@admin_required
def admin_training_progress():
    """Page de progression de l'entraînement"""
    return render_template('training_progress.html')


@app.route('/api/training_status')
@login_required
@admin_required
def api_training_status():
    """API pour obtenir le statut d'entraînement en cours"""
    try:
        # Récupérer le dernier run d'entraînement
        runs = db_manager.get_all_training_runs()
        if not runs:
            return jsonify({"status": "no_training", "message": "Aucun entraînement en cours"})
        
        latest_run = runs[0]
        
        response = {
            "status": latest_run.get("status", "unknown"),
            "started_at": latest_run.get("started_at").isoformat() if latest_run.get("started_at") else None,
            "ended_at": latest_run.get("ended_at").isoformat() if latest_run.get("ended_at") else None,
            "progress": latest_run.get("progress", 0),
            "current_epoch": latest_run.get("current_epoch", 0),
            "total_epochs": latest_run.get("total_epochs", 5),
            "message": latest_run.get("message", ""),
            "used_feedback_count": latest_run.get("used_feedback_count", 0),
            "error_message": latest_run.get("error_message")
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500





@app.route('/admin/download_dataset')
@login_required
@admin_required
def admin_download_dataset():
    """Télécharge le dataset complet (Kaggle + Feedback) sous forme de ZIP"""
    try:
        import shutil
        import kagglehub
        from pathlib import Path
        
        # 1. Localiser le dataset Kaggle
        try:
            # On utilise la même méthode que le trainer pour trouver le chemin
            dataset_path = kagglehub.dataset_download("harshvardhan21/sign-language-detection-using-images")
            dataset_path = Path(dataset_path)
            
            # Trouver le dossier de données réel
            data_dir = None
            for p in dataset_path.rglob('*'):
                if p.is_dir():
                    subdirs = [d for d in p.iterdir() if d.is_dir()]
                    if len(subdirs) > 1:
                        first_class = subdirs[0]
                        if any(first_class.glob('*.jpg')) or any(first_class.glob('*.png')):
                            data_dir = p
                            break
            
            if not data_dir:
                flash("Impossible de localiser le dossier de données source.", "error")
                return redirect(url_for('admin'))
                
        except Exception as e:
            flash(f"Erreur lors de la localisation du dataset: {e}", "error")
            return redirect(url_for('admin'))

        # 2. Créer une archive ZIP temporaire
        # On va créer le zip dans le dossier temporaire du système ou uploads
        zip_filename = f"dataset_sign_language_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
        
        # shutil.make_archive ajoute automatiquement l'extension .zip
        print(f"Création de l'archive {zip_path}.zip depuis {data_dir}...")
        shutil.make_archive(zip_path, 'zip', data_dir)
        
        final_zip_path = f"{zip_path}.zip"
        
        # 3. Envoyer le fichier
        from flask import send_file
        return send_file(
            final_zip_path,
            as_attachment=True,
            download_name=f"{zip_filename}.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        print(f"Erreur download dataset: {e}")
        flash(f"Erreur lors de la création de l'archive: {str(e)}", "error")
        return redirect(url_for('admin'))


# ========== POINT D'ENTRÉE ==========

if __name__ == '__main__':
    print("\nApplication prete!")
    print(f"Database: {app.config['DATABASE_NAME']}")
    print(f"Model classes: {app.config['MODEL_CLASSES']}")
    print(f"Accedez a l'application sur: http://localhost:5000")
    print("=" * 60)
    
    # Environnement de développement : garder debug True mais désactiver le reloader
    # pour éviter certaines erreurs Windows liées au reloader / sockets (WinError 10038).
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
