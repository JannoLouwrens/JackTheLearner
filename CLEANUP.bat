@echo off
echo ============================================================
echo JACK THE WALKER - CLEANUP SCRIPT
echo ============================================================
echo.
echo This will delete:
echo   1. training_data/ folder (259 unnecessary episode files)
echo   2. checkpoints/ folder (old Humanoid-v4 checkpoints)
echo.
echo Why? We upgraded to Humanoid-v5 and removed episode saving
echo.
echo ============================================================
echo.
set /p confirm="Are you sure? Type YES to continue: "

if not "%confirm%"=="YES" (
    echo Cancelled. Nothing was deleted.
    pause
    exit
)

echo.
echo [*] Deleting training_data folder...
if exist training_data (
    rmdir /s /q training_data
    echo [OK] training_data deleted
) else (
    echo [INFO] training_data folder not found
)

echo.
echo [*] Deleting old checkpoints...
if exist checkpoints (
    rmdir /s /q checkpoints
    echo [OK] checkpoints deleted
) else (
    echo [INFO] checkpoints folder not found
)

echo.
echo ============================================================
echo [OK] Cleanup complete!
echo ============================================================
echo.
echo You can now start training from scratch with Humanoid-v5:
echo    py ProgressiveLearning.py
echo.
pause
