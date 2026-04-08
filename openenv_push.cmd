@echo off
REM Helper script to run `openenv push` with UTF-8 encoding on Windows CMD

REM Force console and Python to use UTF-8
chcp 65001 >NUL
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Change to test_env directory relative to this script
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%test_env"

REM Run openenv push for this environment
openenv push --repo-id anshraz27/icu-env
