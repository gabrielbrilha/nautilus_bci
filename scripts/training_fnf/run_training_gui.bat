@echo off
REM Script to launch the BCI Friday Night Funkin' Training Studio GUI on Windows

SET "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\..\..\fnf_prot\python"

uv run python "%SCRIPT_DIR%\train_and_run.py" --gui
pause
