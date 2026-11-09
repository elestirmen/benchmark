@echo off
setlocal
cd /d "%~dp0.."
:: Gradyan buyuklugu (Sobel) taban cizgisi, genis Urgup benchmark'inda Tablo 6 ile ayni 4000 sorgu, yalniz global arama.
:: RAW da ayni sorgularda yeniden hesaplanir (Tablo 6 ile ayni cikmasi, sorgularin ayni oldugunu dogrular).
:: Kesilirse ayni dosyayi yeniden calistirin; kaldigi yerden devam eder.
set "VN_ENV=%USERPROFILE%\anaconda3\envs\visual_navigation_cuda"
set "VN_PY=%VN_ENV%\python.exe"
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate "%VN_ENV%"
set "PYTHONUNBUFFERED=1"
set "TF_CPP_MIN_LOG_LEVEL=2"
set "CUDA_CACHE_PATH=%CD%\outputs\cuda_cache"
set "CUDA_CACHE_MAXSIZE=4294967296"
"%VN_PY%" deney_araclari\sobel_benchmark.py build
if errorlevel 1 (
    echo SOBEL MODELI OLUSTURULAMADI. Bu pencerenin ekran goruntusunu gonderin.
    pause
    exit /b 1
)
"%VN_PY%" run_urgup_cevresi_models.py --model-dir models_sobel --output-dir outputs\sobel_baseline_4000 --max-queries 4000 --search-modes global > outputs\sobel_baseline_4000.log 2>&1
"%VN_PY%" deney_araclari\sobel_benchmark.py summary
echo.
echo Bitti.
pause
endlocal
