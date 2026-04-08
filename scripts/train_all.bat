@echo off
echo ============================================================
echo  TinyBERT - Train on SST-2, MRPC, CoLA
echo ============================================================
call conda activate tinybert

echo.
echo [1/3] Training on SST-2...
python src/train.py --task sst2
echo SST-2 done.

echo.
echo [2/3] Training on MRPC...
python src/train.py --task mrpc
echo MRPC done.

echo.
echo [3/3] Training on CoLA...
python src/train.py --task cola
echo CoLA done.

echo.
echo ============================================================
echo  All tasks complete. Results saved to results/
echo ============================================================
pause
