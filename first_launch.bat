@echo off
echo ==========================================
echo   Installing Japanese Anki Generator...
echo ==========================================

echo [1/5] Creating virtual environment (venv)...
py -3.12 -m venv venv

echo [2/5] Activating venv and installing PyTorch with CUDA 12.6...
call venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo [3/5] Installing other dependencies...
pip install -r requirements.txt

echo [4/5] Loading Models...
python first_launch.py

echo [5/5] Registering model in Ollama...
echo Checking if Ollama is running....
ollama create my-model -f model/Modelfile

echo.
echo ==========================================
echo INSTALLATION COMPLETE! 
echo To start the app, run start.bat
echo ==========================================
pause