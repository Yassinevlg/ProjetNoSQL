Write-Host "Cherche Python 3.11..." -ForegroundColor Cyan
$py311 = "C:\Python311\python.exe"
if (-not (Test-Path $py311)) { $py311 = "$env:ProgramFiles\Python311\python.exe" }
if (-not (Test-Path $py311)) { 
    Write-Host "Python 3.11 non trouvé! Telecharge-le depuis python.org" -ForegroundColor Red
    exit 1 
}
Write-Host "✓ Python 3.11 trouvé" -ForegroundColor Green
$venv_path = "$PSScriptRoot\.venv311"
Write-Host "Creation du venv..." -ForegroundColor Yellow
& $py311 -m venv $venv_path
$activate = "$venv_path\Scripts\Activate.ps1"
& $activate
Write-Host "Installation des paquets..." -ForegroundColor Yellow
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt
Write-Host "✓ Termine!" -ForegroundColor Green
Write-Host "Prochaine fois, active le venv avec: .\.venv311\Scripts\Activate.ps1" -ForegroundColor Cyan
