@echo off
setlocal EnableExtensions
chcp 65001 >nul
for %%I in ("%~dp0.") do set "PROJECT_DIR=%%~fI"
cd /d "%PROJECT_DIR%"

for %%I in ("%PROJECT_DIR%\..") do set "PACKAGE_DIR=%%~fI"
set "PYTHON=%PACKAGE_DIR%\.venv\Scripts\python.exe"
set "HOST=127.0.0.1"
set "PORT=5051"
set "LOCAL_URL=http://%HOST%:%PORT%"
set "UV_LINK_MODE=copy"
set "UV_PROJECT_ENVIRONMENT=%PACKAGE_DIR%\.venv"

set "UV_EXE=%PACKAGE_DIR%\runtime\uv\uv.exe"
if not exist "%UV_EXE%" (
    set "UV_EXE="
    for /f "delims=" %%I in ('where uv 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"
)

if not defined UV_EXE (
    echo [ERROR] uv was not found.
    echo Put uv at Packages\runtime\uv\uv.exe or install it on PATH.
    pause
    exit /b 1
)

if not exist "%PYTHON%" (
    echo [SETUP] Creating shared Packages Python runtime...
    "%UV_EXE%" venv --python 3.12 "%PACKAGE_DIR%\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create the shared Python runtime.
        pause
        exit /b 1
    )
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm was not found on PATH. Install Node.js and try again.
    pause
    exit /b 1
)

powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
    echo [ERROR] Port %PORT% is already in use. Close the existing service and try again.
    pause
    exit /b 2
)

echo [SETUP] Building the web interface...
pushd "%PROJECT_DIR%\frontend"
call npm ci --prefer-offline --no-audit --no-fund
if errorlevel 1 (
    popd
    echo [ERROR] Frontend dependency installation failed.
    pause
    exit /b 1
)
call npm run build
if errorlevel 1 (
    popd
    echo [ERROR] Frontend build failed.
    pause
    exit /b 1
)
popd

echo [CHECK] Checking 识诈 Python dependencies...
call :check_dependencies
if errorlevel 1 (
    echo [SETUP] Installing 识诈 dependencies into the shared runtime...
    "%UV_EXE%" pip install --python "%PYTHON%" -e "%PROJECT_DIR%"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )

    call :check_dependencies
    if errorlevel 1 (
        echo [ERROR] Some 识诈 dependencies are still missing.
        pause
        exit /b 1
    )
)

if /I "%~1"=="--check" (
    echo [OK] The runtime environment is ready.
    exit /b 0
)

echo Starting 识诈 at %LOCAL_URL%...
echo Close this window to stop the service.
start "" /b powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%PROJECT_DIR%\scripts\open_browser_when_ready.ps1" -Url "%LOCAL_URL%"
if exist ".env" (
    "%UV_EXE%" run --no-sync --python "%PYTHON%" --env-file ".env" python "app.py"
) else (
    "%UV_EXE%" run --no-sync --python "%PYTHON%" python "app.py"
)

set "ANTI_FRAUD_EXIT_CODE=%ERRORLEVEL%"

if not "%ANTI_FRAUD_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] 识诈 stopped unexpectedly. Exit code: %ANTI_FRAUD_EXIT_CODE%
    pause
    exit /b %ANTI_FRAUD_EXIT_CODE%
)

exit /b 0

:check_dependencies
"%PYTHON%" -c "import edge_tts, fastapi, httpx, pypinyin, uvicorn, websockets" >nul 2>nul
exit /b %ERRORLEVEL%
