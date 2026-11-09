@echo off
setlocal
cd /d "%~dp0.."
:: Kontrollu karsilastirma: girdi-yeniden-kurma (reconstruction) modeli + harita tarifinin bir tekrari.
:: Egitim + marj kapisi + 1000 sorguluk benchmark, end_to_end_pipeline ile (yaklasik 9 saat).
:: Kayitlar: deneyler\pipeline_<run_id>\kayitlar\
set "VN_ENV=%USERPROFILE%\anaconda3\envs\visual_navigation_cuda"
set "VN_PY=%VN_ENV%\python.exe"
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate "%VN_ENV%"
set "PYTHONUNBUFFERED=1"
echo [recon_kontrol] Python: %VN_PY%
"%VN_PY%" deney_araclari\recon_kontrol.py
echo.
echo Bitti.
pause
endlocal
