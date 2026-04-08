@echo off
echo ============================================================
echo  TinyBERT Environment Setup (Anaconda)
echo  GTX 1660 Ti - CUDA 12.1
echo ============================================================

echo.
echo [1/4] Creating conda environment: tinybert
conda create -n tinybert python=3.10 -y

echo.
echo [2/4] Activating environment
call conda activate tinybert

echo.
echo [3/4] Installing PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [4/4] Installing all other dependencies
pip install -r requirements.txt

echo.
echo ============================================================
echo  Verifying GPU detection...
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not found')"
echo ============================================================

echo.
echo Setup complete! To start training:
echo   conda activate tinybert
echo   python src/train.py --task sst2
echo.
pause
