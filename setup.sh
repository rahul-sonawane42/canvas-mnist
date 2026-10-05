#!/bin/bash

echo "[+] Initializing Canvas-to-Matrix setup..."

if [ ! -d "venv" ]; then
  echo "[+] Creating Python virtual environment (venv)..."
  python3 -m venv venv
fi

echo "[+] Activating virtual environment..."
source venv/bin/activate

echo "[+] Installing NumPy, Pygame, and OpenCV..."
pip install --upgrade pip --quiet
pip install numpy pygame opencv-python --quiet

echo "[+] Dependencies installed successfully."
echo "[+] Launching the inference engine..."

python app.py
