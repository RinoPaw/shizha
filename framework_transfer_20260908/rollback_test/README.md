# 识诈

识诈是一个面向公众的反诈案例检索与对话应用。它把经过脱敏的案例按诈骗类别、风险等级、入口渠道和受害人群组织起来，提供可追溯的案例详情、流式问答与可中断语音播报。

## 能力

- 案例检索：支持关键词、类别、风险等级、入口渠道、受害人群和分页。
- 案例详情：展示诈骗性质、关键手法、风险信号、损失信息、处置建议与来源字段。
- 流式问答：`POST /api/chat` 使用同一套 SearchService 和 SessionStore。
- 连续语音：`/api/voice` 使用讯飞流式 ASR；缺少凭据时会返回明确错误，文本功能不受影响。
- 语音合成：`/api/tts` 使用 Edge TTS，音色、语速和音调均可由环境变量设置。

## 快速开始

要求 Node.js 22+、Python 3.12+。在 `Packages` 目录中运行：

```powershell
cd D:\Projects\Packages\识诈
Copy-Item .env.example .env
.\start.bat
```

启动器优先复用上级 `Packages/.venv` 与 `Packages/runtime/uv/uv.exe`，首次启动会构建前端，并在服务就绪后打开 `http://127.0.0.1:5051`。应用不会自行读取 `.env`；启动器通过环境文件把配置传给进程。

没有配置 `AI_API_KEY` 时，文字问答仍可使用本地降级回答。连续语音需要同时配置 `XF_APP_ID`、`XF_API_KEY`、`XF_API_SECRET`。

## 配置

完整示例见 `.env.example`。主要变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | 服务监听地址 |
| `PORT` | `5051` | 服务端口 |
| `DATASET_PATH` | `data/processed/case_items.json` | 反诈案例数据 |
| `AI_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容聊天接口 |
| `AI_MODEL` | `deepseek-v4-flash` | 聊天模型 |
| `TTS_VOICE` | `zh-CN-XiaoxiaoNeural` | Edge TTS 音色 |
| `TTS_RATE` | `-2%` | Edge TTS 语速 |
| `TTS_PITCH` | `+0Hz` | Edge TTS 音调 |

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 健康检查 |
| GET | `/api/meta` | 数据规模、类别、风险等级、入口渠道与能力 |
| GET | `/api/categories` | 类别及数量 |
| GET | `/api/items` | `q`、`category`、`risk_level`、`entry_channel`、`victim_group` 检索 |
| GET | `/api/items/{case_id}` | 完整案例详情 |
| POST | `/api/chat` | SSE 流式问答 |
| POST | `/api/chat/{session_id}/turn/{turn_id}/cancel` | 取消回答 |
| WS | `/api/voice` | 讯飞流式 ASR 与连续对话 |
| GET | `/api/tts` | Edge TTS 音频流 |

## Docker / Render

```powershell
docker build -t anti-fraud-explorer .
docker run --rm -p 5051:5051 --env-file .env -e HOST=0.0.0.0 anti-fraud-explorer
```

`render.yaml` 使用 Docker 部署并以 `/healthz` 作为健康检查。

## 安全

- 不提交 `.env`、API 密钥、日志或本地 embedding 索引。
- 案例已做脱敏；公开部署前仍应复核来源与内容。
- 反诈建议不能替代警方、银行或其他专业机构的正式处置意见。
