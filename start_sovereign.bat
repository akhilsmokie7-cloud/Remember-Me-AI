@echo off
title Sovereign Mind - V1.0 Console

echo [?] check: Verifying Sovereign Stack...
python install_brain.py
if %errorlevel% neq 0 (
    echo [!] Critical Error in Bootloader.
    pause
    exit
)

echo [1/3] Ignition: Starting Local Inference Engine...
start /min "Sovereign Engine" "brain_engine\llama-server.exe" -m "brain_engine\model.gguf" -c 8192 --port 8081 --n-gpu-layers 35

echo [2/3] Waiting for Engine warmup...
timeout /t 5 /nobreak >nul

echo [3/3] Launching Cognitive Interface...
python -m streamlit run src/interface.py --server.port 8503 --server.headless true

pause
