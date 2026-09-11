@echo off
setlocal EnableExtensions
chcp 65001 >nul

for %%I in ("%~dp0.") do set "PROJECT_DIR=%%~fI"
cd /d "%PROJECT_DIR%" || exit /b 1

set "HOST=127.0.0.1"
set "PORT=5051"
set "LOCAL_URL=http://%HOST%:%PORT%"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "UV_LINK_MODE=copy"
set "UV_PROJECT_ENVIRONMENT=%VENV_DIR%"
set "CHECK_ONLY=0"
set "NO_BROWSER=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--check" set "CHECK_ONLY=1"
if /I "%~1"=="--no-browser" set "NO_BROWSER=1"
shift
goto parse_args

:args_done
echo.
echo ========================================
echo   识诈 Windows 启动器
echo ========================================
echo 项目目录: %PROJECT_DIR%
echo 服务地址: %LOCAL_URL%
echo.

call :find_uv
if errorlevel 1 exit /b 1

call :ensure_node
if errorlevel 1 exit /b 1

call :ensure_python
if errorlevel 1 exit /b 1

echo [SETUP] 安装/同步 Python 依赖...
"%UV_EXE%" pip install --python "%PYTHON%" -e "%PROJECT_DIR%"
if errorlevel 1 (
    echo [ERROR] Python 依赖安装失败。
    exit /b 1
)

echo [SETUP] 安装前端依赖并构建...
pushd "%PROJECT_DIR%\frontend"
call npm ci --prefer-offline --no-audit --no-fund
if errorlevel 1 (
    popd
    echo [ERROR] 前端依赖安装失败。
    exit /b 1
)
call npm run build
if errorlevel 1 (
    popd
    echo [ERROR] 前端构建失败。
    exit /b 1
)
popd

if "%CHECK_ONLY%"=="1" (
    echo [OK] 识诈运行环境已准备完成。
    exit /b 0
)

call :free_port
if errorlevel 1 exit /b 1

if "%NO_BROWSER%"=="0" (
    if exist "%PROJECT_DIR%\scripts\open_browser_when_ready.ps1" (
        start "" /b powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%PROJECT_DIR%\scripts\open_browser_when_ready.ps1" -Url "%LOCAL_URL%"
    )
)

echo.
echo [START] 启动识诈：%LOCAL_URL%
echo [INFO] 关闭此窗口或按 Ctrl+C 可停止服务。
echo.

if exist "%PROJECT_DIR%\.env" (
    "%UV_EXE%" run --no-sync --python "%PYTHON%" --env-file "%PROJECT_DIR%\.env" python "%PROJECT_DIR%\app.py"
) else (
    echo [WARN] 未找到 .env；云端模型和实时语音可能不可用。
    "%UV_EXE%" run --no-sync --python "%PYTHON%" python "%PROJECT_DIR%\app.py"
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] 识诈异常退出，退出码：%EXIT_CODE%
)
exit /b %EXIT_CODE%

:find_uv
set "UV_EXE="

if exist "%PROJECT_DIR%\runtime\uv\uv.exe" set "UV_EXE=%PROJECT_DIR%\runtime\uv\uv.exe"
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV_EXE (
    for /f "delims=" %%I in ('where uv 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"
)

if defined UV_EXE (
    echo [OK] uv: %UV_EXE%
    exit /b 0
)

echo [SETUP] 未找到 uv，正在安装...
set "UV_INSTALLER=%TEMP%\uv-install-%RANDOM%.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest 'https://astral.sh/uv/install.ps1' -OutFile '%UV_INSTALLER%' -UseBasicParsing"
if errorlevel 1 (
    echo [ERROR] 无法下载 uv 安装脚本。
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%UV_INSTALLER%"
del /q "%UV_INSTALLER%" >nul 2>nul

if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV_EXE (
    for /f "delims=" %%I in ('where uv 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"
)
if not defined UV_EXE (
    echo [ERROR] uv 安装完成后仍无法找到 uv.exe。
    exit /b 1
)

echo [OK] uv: %UV_EXE%
exit /b 0

:ensure_node
set "NODE_MAJOR="
where node >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%I in ('node -p "process.versions.node.split('.')[0]" 2^>nul') do set "NODE_MAJOR=%%I"
)

if defined NODE_MAJOR (
    powershell.exe -NoProfile -Command "if ([int]'%NODE_MAJOR%' -ge 22) { exit 0 } else { exit 1 }"
    if not errorlevel 1 (
        echo [OK] Node.js 已就绪。
        where npm >nul 2>nul
        if not errorlevel 1 exit /b 0
    )
)

echo [SETUP] 需要 Node.js 22+，尝试通过 winget 安装 Node.js LTS...
where winget >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 未找到可用的 Node.js 22+，且本机没有 winget。
    echo         请先安装 Node.js 22 LTS 后重新运行 start.bat。
    exit /b 1
)

winget install --id OpenJS.NodeJS.LTS -e --accept-package-agreements --accept-source-agreements
set "PATH=%ProgramFiles%\nodejs;%PATH%"
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js 安装后仍不在 PATH，请重新打开终端后再运行 start.bat。
    exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm 不可用，请重新打开终端后再运行 start.bat。
    exit /b 1
)

echo [OK] Node.js 已就绪。
exit /b 0

:ensure_python
if exist "%PYTHON%" (
    "%PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 (
        echo [OK] Python 3.12 虚拟环境已存在。
        exit /b 0
    )
    echo [SETUP] 现有虚拟环境不可用，正在重建...
    rmdir /s /q "%VENV_DIR%" >nul 2>nul
)

echo [SETUP] 创建项目本地 Python 3.12 环境...
"%UV_EXE%" venv --python 3.12 "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Python 3.12 虚拟环境创建失败。
    exit /b 1
)
if not exist "%PYTHON%" (
    echo [ERROR] 未找到虚拟环境 Python：%PYTHON%
    exit /b 1
)

echo [OK] Python 3.12 已就绪。
exit /b 0

:free_port
set "PORT_IN_USE=0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ids = @(Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique); if ($ids.Count -eq 0) { exit 0 }; foreach ($procId in $ids) { Write-Host ('[STOP] 端口 %PORT% 正被 PID ' + $procId + ' 占用，正在停止...'); try { Stop-Process -Id $procId -Force -ErrorAction Stop } catch { Write-Host ('[ERROR] 无法停止 PID ' + $procId); exit 1 } }; Start-Sleep -Milliseconds 400; if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
    echo [ERROR] 无法释放端口 %PORT%。
    exit /b 1
)
exit /b 0
