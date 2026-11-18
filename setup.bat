@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment. Install Python 3.10 or newer and try again.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Put the two dataset files in the datasets folder, then run:
echo   .venv\Scripts\python.exe -m src.train_models
echo   .venv\Scripts\python.exe -m streamlit run app.py
endlocal
