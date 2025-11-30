Write-Host "Téléchargement de Python 3.11..." -ForegroundColor Cyan
$url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$installer = "$env:TEMP\python311.exe"
$progressPreference = 'silentlyContinue'
Invoke-WebRequest -Uri $url -OutFile $installer
Write-Host "✓ Téléchargé" -ForegroundColor Green
Write-Host "Installation de Python 3.11..." -ForegroundColor Cyan
& $installer /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
Write-Host "✓ Installation terminée!" -ForegroundColor Green
Remove-Item $installer -Force
Write-Host "Vérification..." -ForegroundColor Yellow
py -3.11 --version
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python 3.11 est prêt!" -ForegroundColor Green
} else {
    Write-Host "✗ Erreur - Relance le script avec admin" -ForegroundColor Red
}
