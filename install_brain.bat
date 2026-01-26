@echo off
setlocal
title Sovereign Mind - Brain Installation

echo ===================================================
echo   SOVEREIGN MIND - INTELLIGENCE DOWNLOADER
echo   Deploying Local AI... (0g Cloud Reliance)
echo ===================================================

if not exist "brain_engine" mkdir "brain_engine"

rem 1. Download Llama.cpp Server (Lightweight Inference Engine)
if not exist "brain_engine\llama-server.exe" (
    echo [1/3] Downloading Inference Engine (llama-server)...
    echo        Source: GitHub Releases (b3600)
    curl -L -o brain_engine\llama-server.zip https://github.com/ggerganov/llama.cpp/releases/download/b3600/llama-b3600-bin-win-avx2-x64.zip
    
    echo [1/3] Extracting Engine...
    tar -xf brain_engine\llama-server.zip -C brain_engine
    del brain_engine\llama-server.zip
) else (
    echo [1/3] Engine already present.
)

rem 2. Download Qwen 2.5 Coder 7B (The Brain)
if not exist "brain_engine\model.gguf" (
    echo [2/3] Downloading Qwen 2.5 Coder 7B (SOTA Compressed Model)...
    echo        Size: ~4.7 GB. This may take a while depending on your internet.
    echo        Source: HuggingFace (Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)
    
    rem Direct link to the Q4_K_M quantized model (Best balance of speed/quality)
    curl -L -o brain_engine\model.gguf https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf?download=true
) else (
    echo [2/3] Brain model already present.
)

echo [3/3] Configuration Complete.
echo ===================================================
echo   INSTALLATION SUCCESSFUL.
echo   You can now run 'start_sovereign.bat'
echo ===================================================
pause
