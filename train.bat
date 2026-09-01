@echo off
title Train Local Summarization Model
cls

echo ========================================================
echo       Starting Local Model Training
echo ========================================================
echo.

echo [*] Training t5-small on cnn_dailymail dataset...
echo [*] Checkpoints saved EVERY epoch
echo [*] Auto-resumes from latest checkpoint if found
echo.

cd /d "%~dp0training"

:: Check for latest checkpoint automatically
set RESUME_ARG=
if exist "checkpoints" (
    :: Find the highest epoch checkpoint
    set LATEST=0
    for /D %%d in (checkpoints\epoch-*) do (
        set LATEST=%%d
    )
    if not "!LATEST!"=="0" (
        set RESUME_ARG=--resume_from !LATEST!
        echo [*] Found checkpoint: !LATEST!
    )
)

python summarizer_train.py ^
    --model_name t5-small ^
    --dataset ..\data ^
    --per_device_train_batch_size 2 ^
    --per_device_eval_batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --fp16 ^
    --num_train_epochs 3 ^
    --save_epochs 1 ^
    --max_train_samples 5000 ^
    --max_eval_samples 500 ^
    --checkpoint_dir checkpoints ^
    --output_dir ..\models ^
    --logging_steps 25 ^
    --eval_steps 500

cd /d "%~dp0"

echo.
echo [*] Training finished! Check "models\" for the final model.
pause
