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

# Initialiser et charger le modèle CNN
cnn_model = CNNModelLoader(
    model_path=app.config['MODEL_PATH'],
    classes=app.config['MODEL_CLASSES'],
    input_size=app.config['MODEL_INPUT_SIZE'],
    grayscale=app.config['MODEL_GRAYSCALE']
)

# Charger le modèle au démarrage
# Charger le modèle au démarrage
print("=" * 60)
print("Demarrage de l'application CNN + MongoDB")
print("=" * 60)
cnn_model.load_model()
print("=" * 60)


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
        # Astuce: Si c'est le tout premier utilisateur, on peut le mettre admin
        # Mais restons simple pour l'instant
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
    """Page d'accueil"""
    try:
        # Récupérer quelques statistiques pour la page d'accueil
        total_predictions = db_manager.get_total_predictions()
        model_info = cnn_model.get_model_info()
        
        return render_template(
            'index.html',
            total_predictions=total_predictions,
            model_info=model_info
        )
    except Exception as e:
        print(f"Erreur dans index: {e}")
        return render_template('index.html', total_predictions=0, model_info={})


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
            user_id=session.get('user_id'), # Lier à l'utilisateur
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


@app.route('/statistics')
@admin_required
def statistics():
    """Page affichant les statistiques et agrégations MongoDB"""
    try:
        # Récupérer toutes les statistiques
        stats = db_manager.get_global_statistics()
        confusion_data = db_manager.get_confusion_data()
        true_labels = db_manager.get_true_labels_distribution()
        
        return render_template(
            'statistics.html',
            stats=stats,
            confusion_data=confusion_data,
            true_labels=true_labels
        )
        
    except Exception as e:
        flash(f'Erreur lors de la récupération des statistiques: {str(e)}', 'error')
        print(f"Erreur statistics: {e}")
        return render_template('statistics.html', stats={}, confusion_data=[], true_labels=[])


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
        users = db_manager.get_all_users()
        total_predictions = db_manager.get_total_predictions()
        active_users_count = sum(1 for u in users if u.get('is_active'))

        return render_template(
            'admin.html',
            users=users,
            total_predictions=total_predictions,
            active_users_count=active_users_count
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
