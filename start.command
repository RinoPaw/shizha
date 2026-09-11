#!/bin/bash

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1

HOST="127.0.0.1"
PORT="5051"
LOCAL_URL="http://${HOST}:${PORT}"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
export UV_LINK_MODE="copy"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

BROWSER_WATCH_PID=""

pause_error() {
    local code="${1:-1}"
    echo
    read -r -p "按回车键关闭窗口..." _
    exit "$code"
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    if [ -x "$HOME/.local/bin/uv" ]; then
        printf '%s\n' "$HOME/.local/bin/uv"
        return 0
    fi
    return 1
}

is_running() {
    curl -fsS --max-time 1 "$LOCAL_URL/healthz" >/dev/null 2>&1
}

port_pid() {
    lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1
}

open_browser() {
    local app="/Applications/Google Chrome for Testing.app"
    local downloaded="$HOME/Downloads/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"

    if [ -d "$app" ]; then
        open -a "Google Chrome for Testing" "$LOCAL_URL" >/dev/null 2>&1
    elif [ -x "$downloaded" ]; then
        "$downloaded" --new-window "$LOCAL_URL" >/dev/null 2>&1 &
    else
        open "$LOCAL_URL" >/dev/null 2>&1
    fi
}

open_browser_when_ready() {
    local i=0
    while [ "$i" -lt 80 ]; do
        if is_running; then
            open_browser
            return 0
        fi
        sleep 0.25
        i=$((i + 1))
    done
    return 1
}

show_running_menu() {
    while true; do
        echo
        echo "识诈已经在运行：$LOCAL_URL"
        echo "[O] 打开浏览器   [S] 停止识诈   [Q] 退出"
        read -r -p "> " choice
        case "$choice" in
            O|o)
                open_browser
                ;;
            S|s)
                pid="$(port_pid)"
                if [ -n "$pid" ]; then
                    echo "[STOP] 正在停止 PID $pid ..."
                    kill "$pid" 2>/dev/null || true
                    sleep 0.5
                    if is_running; then
                        echo "[ERROR] 识诈仍在运行。"
                    else
                        echo "[OK] 识诈已停止。"
                        return 0
                    fi
                else
                    echo "[WARN] 没找到监听 $PORT 的进程。"
                fi
                ;;
            Q|q)
                exit 0
                ;;
            *)
                echo "请输入 O、S 或 Q。"
                ;;
        esac
    done
}

wait_for_free_port() {
    while true; do
        if is_running; then
            show_running_menu
        fi

        pid="$(port_pid)"
        if [ -z "$pid" ]; then
            return 0
        fi

        echo
        echo "[WARN] 端口 $PORT 已被占用，但识诈没有响应。"
        ps -p "$pid" -o pid=,comm= 2>/dev/null || echo "PID $pid"
        echo "[R] 重新检查   [S] 停止该进程   [Q] 退出"
        read -r -p "> " choice
        case "$choice" in
            R|r)
                ;;
            S|s)
                echo "[STOP] 正在停止 PID $pid ..."
                if ! kill "$pid" 2>/dev/null; then
                    echo "[ERROR] 无法停止 PID $pid。"
                fi
                sleep 0.5
                ;;
            Q|q)
                exit 0
                ;;
            *)
                echo "请输入 R、S 或 Q。"
                ;;
        esac
    done
}

cleanup() {
    if [ -n "$BROWSER_WATCH_PID" ]; then
        kill "$BROWSER_WATCH_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

clear
echo "========================================"
echo "  识诈 macOS 启动器"
echo "========================================"
echo "项目目录: $PROJECT_DIR"
echo "服务地址: $LOCAL_URL"
echo

if is_running; then
    show_running_menu
fi

UV_EXE="$(find_uv || true)"
if [ -z "$UV_EXE" ]; then
    echo "[ERROR] 未找到 uv。"
    echo "请先安装 uv，或确保 ~/.local/bin/uv 可用。"
    pause_error 1
fi

echo "[OK] uv: $UV_EXE"

if [ ! -x "$PYTHON" ]; then
    echo "[SETUP] 创建 Python 3.12 虚拟环境..."
    if ! "$UV_EXE" venv --python 3.12 "$VENV_DIR"; then
        echo "[ERROR] Python 3.12 虚拟环境创建失败。"
        pause_error 1
    fi
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "[ERROR] 未找到 npm，请先安装 Node.js 22+。"
    pause_error 1
fi

echo "[SETUP] 构建前端..."
cd "$PROJECT_DIR/frontend" || pause_error 1
if ! npm ci --prefer-offline --no-audit --no-fund; then
    echo "[ERROR] 前端依赖安装失败。"
    pause_error 1
fi
if ! npm run build; then
    echo "[ERROR] 前端构建失败。"
    pause_error 1
fi
cd "$PROJECT_DIR" || pause_error 1

echo "[CHECK] 检查 Python 依赖..."
if ! "$PYTHON" -c "import edge_tts, fastapi, httpx, pypinyin, uvicorn, websockets" >/dev/null 2>&1; then
    echo "[SETUP] 安装识诈 Python 依赖..."
    if ! "$UV_EXE" pip install --python "$PYTHON" -e "$PROJECT_DIR"; then
        echo "[ERROR] Python 依赖安装失败。"
        pause_error 1
    fi
fi

wait_for_free_port

open_browser_when_ready &
BROWSER_WATCH_PID=$!

echo
echo "[START] 启动识诈：$LOCAL_URL"
echo "[INFO] 日志会持续显示在此窗口。按 Ctrl+C 即可停止。"
echo

if [ -f "$PROJECT_DIR/.env" ]; then
    "$UV_EXE" run --no-sync --python "$PYTHON" --env-file "$PROJECT_DIR/.env" python "$PROJECT_DIR/app.py"
else
    echo "[WARN] 未找到 .env；云端模型和实时语音可能不可用。"
    "$UV_EXE" run --no-sync --python "$PYTHON" python "$PROJECT_DIR/app.py"
fi

EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 130 ]; then
    echo
    echo "[STOP] 识诈已停止。"
    exit 0
fi

if [ "$EXIT_CODE" -ne 0 ]; then
    echo
    echo "[ERROR] 识诈异常退出，退出码：$EXIT_CODE"
    pause_error "$EXIT_CODE"
fi

echo
echo "[STOP] 识诈已停止。"
