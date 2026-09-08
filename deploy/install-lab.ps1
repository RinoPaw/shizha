[CmdletBinding()]
param(
    [string]$InstallRoot = "D:\Projects\Packages",
    [string]$Repository = "https://github.com/RinoPaw/shizha.git",
    [string]$Branch = "main",
    [string]$EnvFile = "",
    [switch]$SkipModels,
    [switch]$SkipLaunch
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

function Step([string]$Text) { Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Require-Command([string]$Name, [string]$WingetId) {
    if (Get-Command $Name -ErrorAction SilentlyContinue) { return }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "缺少 $Name，且本机没有 winget。请先安装 $Name 后重试。"
    }
    Step "安装 $Name"
    winget install --id $WingetId -e --accept-package-agreements --accept-source-agreements
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw "$Name 安装后仍不在 PATH。请重开终端再运行。" }
}

function Get-Uv([string]$Root) {
    $bundled = Join-Path $Root "runtime\uv\uv.exe"
    if (Test-Path -LiteralPath $bundled) { return $bundled }
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    Step "安装 uv"
    $installer = Join-Path $env:TEMP "uv-install.ps1"
    Invoke-WebRequest "https://astral.sh/uv/install.ps1" -OutFile $installer -UseBasicParsing
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer
    $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path -LiteralPath $uv)) { throw "uv 安装失败。" }
    return $uv
}

function Download-Models([string]$Uv, [string]$Project, [string]$ModelsRoot) {
    $manifestPath = Join-Path $Project "deploy\models.manifest.json"
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    New-Item -ItemType Directory -Force -Path $ModelsRoot | Out-Null
    foreach ($model in $manifest.models) {
        $destination = Join-Path $ModelsRoot $model.destination
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        Step "Hugging Face：$($model.name)"
        $args = @("--from", "huggingface_hub", "hf", "download", $model.repo_id,
                  "--revision", $model.revision, "--local-dir", $destination)
        foreach ($pattern in $model.files) { $args += @("--include", $pattern) }
        & $Uv tool run @args
        if ($LASTEXITCODE -ne 0) { throw "模型下载失败：$($model.name)" }
        foreach ($property in $model.sha256.PSObject.Properties) {
            $file = Join-Path $destination $property.Name
            if (-not (Test-Path -LiteralPath $file)) { throw "模型文件缺失：$file" }
            $actual = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne [string]$property.Value) { throw "模型校验失败：$file" }
        }
    }
}

Step "检查基础工具"
Require-Command "git" "Git.Git"
Require-Command "node" "OpenJS.NodeJS.LTS"
Require-Command "npm" "OpenJS.NodeJS.LTS"

$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$Project = Join-Path $InstallRoot "识诈"
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

Step "同步 GitHub 源码"
if (Test-Path -LiteralPath (Join-Path $Project ".git")) {
    git -C $Project fetch origin $Branch --prune
    git -C $Project checkout $Branch
    git -C $Project pull --ff-only origin $Branch
} elseif (Test-Path -LiteralPath $Project) {
    throw "目标目录已存在但不是 Git 仓库：$Project"
} else {
    git clone --branch $Branch --single-branch $Repository $Project
}
if ($LASTEXITCODE -ne 0) { throw "GitHub 源码同步失败。" }

$Uv = Get-Uv $InstallRoot
if (-not $SkipModels) { Download-Models $Uv $Project (Join-Path $InstallRoot "models") }

if (-not $EnvFile) {
    $beside = Join-Path (Split-Path -Parent $PSCommandPath) "shizha.env"
    if (Test-Path -LiteralPath $beside) { $EnvFile = $beside }
}
if ($EnvFile) {
    $resolvedEnv = (Resolve-Path -LiteralPath $EnvFile).Path
    Copy-Item -LiteralPath $resolvedEnv -Destination (Join-Path $Project ".env") -Force
} elseif (-not (Test-Path -LiteralPath (Join-Path $Project ".env"))) {
    Copy-Item -LiteralPath (Join-Path $Project ".env.example") -Destination (Join-Path $Project ".env")
    Write-Warning "已生成 .env；云端模型和讯飞语音需要填入密钥。"
}

Step "安装依赖并构建"
& (Join-Path $Project "start.bat") --check
if ($LASTEXITCODE -ne 0) { throw "识诈运行环境检查失败。" }

Step "安装完成"
Write-Host "源码：$Project"
Write-Host "模型：$(Join-Path $InstallRoot 'models')"
Write-Host "启动：$(Join-Path $Project 'start.bat')"
if (-not $SkipLaunch) { & (Join-Path $Project "start.bat") }
