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

rem Like the macOS launcher: if 识诈 is already healthy, just open it.
if /I not "%~1"=="--check" (
    call :is_running
    if not errorlevel 1 (
        echo [INFO] 识诈已经在运行，正在打开浏览器...
        start "" "%LOCAL_URL%"
        exit /b 0
    )
)

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

call :wait_for_port
set "PORT_RESULT=%ERRORLEVEL%"
if "%PORT_RESULT%"=="10" exit /b 0
if not "%PORT_RESULT%"=="0" exit /b %PORT_RESULT%

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

:is_running
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%LOCAL_URL%/healthz' -TimeoutSec 1; if ($response.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>nul
exit /b %ERRORLEVEL%

:wait_for_port
call :is_running
if not errorlevel 1 (
    echo [INFO] 识诈已经在运行，正在打开浏览器...
    start "" "%LOCAL_URL%"
    exit /b 10
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$connections = @(Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue); if ($connections.Count -eq 0) { exit 0 }; Write-Host '[WARN] 端口 %PORT% 已被占用：'; $ids = @($connections | Select-Object -ExpandProperty OwningProcess -Unique); foreach ($ownerId in $ids) { $process = Get-Process -Id $ownerId -ErrorAction SilentlyContinue; if ($process) { Write-Host ('       PID {0}  {1}' -f $ownerId, $process.ProcessName) } else { Write-Host ('       PID {0}' -f $ownerId) } }; exit 1"
if not errorlevel 1 exit /b 0

echo.
echo [WARN] 5051 端口被其他程序占用，但识诈当前没有响应。
echo        请关闭上面显示的程序，或处理旧的识诈进程后再试。
echo.
choice /C RQ /N /M "[R] 重新检查  [Q] 退出: "
if errorlevel 2 exit /b 2
goto wait_for_port

:check_dependencies
"%PYTHON%" -c "import edge_tts, fastapi, httpx, pypinyin, uvicorn, websockets" >nul 2>nul
exit /b %ERRORLEVEL%
