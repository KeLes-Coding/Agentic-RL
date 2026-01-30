@echo off
chcp 65001 >nul
REM Sync remote logs and data to local (Windows)
REM Usage: sync_logs.bat

set REMOTE_HOST=zzh@218.192.110.198
set REMOTE_BASE=~/Workspace/260126/agentic-rl
set LOCAL_BASE=.

echo ================================================
echo    CCAPO Log Sync Tool (Windows)
echo ================================================
echo Remote: %REMOTE_HOST%:%REMOTE_BASE%
echo Local:  %LOCAL_BASE%
echo.

REM Create local directories if not exist
if not exist "%LOCAL_BASE%\logger" mkdir "%LOCAL_BASE%\logger"
if not exist "%LOCAL_BASE%\local_logger" mkdir "%LOCAL_BASE%\local_logger"
if not exist "%LOCAL_BASE%\stdb" mkdir "%LOCAL_BASE%\stdb"

REM Sync logger directory
echo -------------------------------------------
echo Syncing: logger/
scp -r %REMOTE_HOST%:%REMOTE_BASE%/logger/* %LOCAL_BASE%\logger\
if %ERRORLEVEL% == 0 (
    echo   [OK] logger/
) else (
    echo   [SKIP] logger/ - empty or connection issue
)

REM Sync local_logger directory
echo -------------------------------------------
echo Syncing: local_logger/
scp -r %REMOTE_HOST%:%REMOTE_BASE%/local_logger/* %LOCAL_BASE%\local_logger\
if %ERRORLEVEL% == 0 (
    echo   [OK] local_logger/
) else (
    echo   [SKIP] local_logger/ - empty or connection issue
)

REM Sync stdb directory
echo -------------------------------------------
echo Syncing: stdb/
scp -r %REMOTE_HOST%:%REMOTE_BASE%/stdb/* %LOCAL_BASE%\stdb\
if %ERRORLEVEL% == 0 (
    echo   [OK] stdb/
) else (
    echo   [SKIP] stdb/ - empty or connection issue
)

echo.
echo ================================================
echo Sync Complete!
echo.
echo Run analysis: python analyze_full_trace.py
echo ================================================
pause
