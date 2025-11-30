# ============================================================================
# Script PowerShell : Setup Conda + TensorFlow + Notebook d'entraînement CNN
# ============================================================================
# Ce script :
# 1) Vérifie si Conda est installé (si non, fournit instruction)
# 2) Crée un environnement Conda Python 3.11 (cnn-py311)
# 3) Active l'environnement
# 4) Installe TensorFlow 2.13.0 + Jupyter + dépendances du projet
# 5) Lance le notebook d'entraînement du modèle CNN
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Conda + TensorFlow pour CNN" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# ============================================================================
# 1) Vérifier que Conda est installé
# ============================================================================
Write-Host "[1/5] Vérification de Conda..." -ForegroundColor Yellow

try {
    $condaVersion = conda --version 2>$null
    Write-Host "✅ Conda trouvé: $condaVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Conda n'est pas trouvé. Veuillez installer Miniconda depuis:" -ForegroundColor Red
    Write-Host "   https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Cyan
    Write-Host "`nAprès installation, relancez ce script." -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# 2) Créer environnement Conda Python 3.11
# ============================================================================
Write-Host "`n[2/5] Création de l'environnement Conda (cnn-py311, Python 3.11)..." -ForegroundColor Yellow
Write-Host "      (Cela peut prendre quelques minutes)" -ForegroundColor Gray

conda create -n cnn-py311 python=3.11 -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors de la création de l'environnement Conda." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Environnement Conda créé." -ForegroundColor Green

# ============================================================================
# 3) Activer l'environnement Conda
# ============================================================================
Write-Host "`n[3/5] Activation de l'environnement cnn-py311..." -ForegroundColor Yellow

# Initialiser conda pour PowerShell
conda init powershell --no-modify-path | Out-Null
conda activate cnn-py311

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Tentative d'activation alternative..." -ForegroundColor Yellow
    # Fallback: dot-source le script d'activation
    & "$(conda info --base)\Scripts\Activate.ps1" cnn-py311
}

Write-Host "✅ Environnement activé." -ForegroundColor Green

# ============================================================================
# 4) Installer TensorFlow + Jupyter + dépendances
# ============================================================================
Write-Host "`n[4/5] Installation de TensorFlow 2.13.0 + Jupyter + dépendances..." -ForegroundColor Yellow
Write-Host "      (Cela peut prendre 5-10 minutes)" -ForegroundColor Gray

# Mettre pip à jour
python -m pip install --upgrade pip -q

# Installer TensorFlow et Jupyter
pip install tensorflow==2.13.0 jupyter -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors de l'installation de TensorFlow." -ForegroundColor Red
    exit 1
}

# Installer les dépendances du projet
$requirementsPath = "$PSScriptRoot\requirements.txt"
if (Test-Path $requirementsPath) {
    pip install -r $requirementsPath -q
    Write-Host "✅ TensorFlow, Jupyter et dépendances installés." -ForegroundColor Green
}
else {
    Write-Host "⚠️  requirements.txt non trouvé, installation partielle." -ForegroundColor Yellow
}

# ============================================================================
# 5) Lancer le notebook d'entraînement
# ============================================================================
Write-Host "`n[5/5] Lancement du notebook d'entraînement..." -ForegroundColor Yellow
Write-Host "      (Une fenêtre Jupyter s'ouvrira dans votre navigateur)" -ForegroundColor Gray

$notebookPath = "$PSScriptRoot\train_sign_language_cnn.ipynb"
if (Test-Path $notebookPath) {
    jupyter notebook $notebookPath
}
else {
    Write-Host "⚠️  Notebook non trouvé: $notebookPath" -ForegroundColor Yellow
    Write-Host "   Vous pouvez le lancer manuellement avec:" -ForegroundColor Gray
    Write-Host "   jupyter notebook train_sign_language_cnn.ipynb" -ForegroundColor Cyan
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "✅ Setup terminé !" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
