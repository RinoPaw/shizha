# 识诈 (Anti-Fraud Explorer)

> 面向反诈宣传、案例检索、智能问答与内容转化的本地 Web 智能体。

**识诈** 是一个基于 Flask 的本地化反诈知识库应用，内置案例检索引擎、大模型问答管道和多场景内容生成能力，支持服务端语音合成（TTS）。无需复杂部署，配置 API Key 后即可运行。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **案例检索** | 关键词 + 拼音模糊匹配 + 可选向量语义检索，支持分类/风险等级/渠道多维度筛选 |
| **智能问答** | 基于案例库做依据式回答，模型与本地检索协作，不命中时明确说明资料边界 |
| **场景推荐** | 按校园、社区、企业、老年等宣讲场景匹配更合适的案例 |
| **内容转化** | 将案例改写为讲解词、学习任务、展示方案、年轻化文案等 |
| **语音播报** | 支持火山引擎 TTS（v1/v3）和 OpenAI 兼容接口，失败时自动回退浏览器语音 |

---

## 技术栈

- **后端**: Python 3.14 + Flask
- **AI 推理**: OpenAI 兼容 API / 智谱 AI（自动路由）
- **检索**: 词法评分 + 可选 Embedding 语义融合（RRF）
- **语音**: 火山引擎 TTS + OpenAI TTS 回退
- **配置**: Pydantic Settings（`.env` + 环境变量）

---

## 快速开始

### 环境要求

- Python `>= 3.14`
- Windows / Linux / macOS

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv pip install -e ".[dev]"

# 或使用 pip
python -m pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入核心配置：

```env
# 必填：AI 模型
AI_API_KEY=your-api-key
AI_BASE_URL=https://api.deepseek.com
AI_MODEL=deepseek-v4-flash

# 可选：语义检索
EMBEDDING_API_KEY=your-embedding-key
SEARCH_USE_EMBEDDING=true

# 可选：语音合成（火山引擎）
VOLC_TTS_API_KEY=your-volc-key
VOLC_TTS_APP_ID=your-app-id
```

完整配置项参见 `.env.example`。

### 3. 准备数据

仓库已包含可直接运行的规范化数据集：

```
data/processed/case_items.json
```

如需从原始数据重新生成：

```bash
python scripts/process_raw_data.py
```

### 4. 构建嵌入索引（可选）

启用语义检索前，先执行：

```bash
python scripts/maintenance/rebuild_embedding_index.py
```

### 5. 启动服务

```bash
python app.py
```

访问 [http://127.0.0.1:5070](http://127.0.0.1:5070)

---

## 开发

```bash
# 运行测试
python -m pytest -q

# 代码检查
python -m ruff check .

# 自动格式化
python -m ruff format .
```

---

## 项目结构

```text
.
├── app.py                          # 开发入口
├── pyproject.toml                  # 项目配置与依赖
├── data/
│   ├── raw/
│   │   └── dataset.json            # 原始案例数据
│   ├── processed/
│   │   └── case_items.json         # 规范化数据集（schema v4）
│   └── embeddings/
│       └── case_embeddings.json    # 语义检索索引
├── scripts/
│   ├── process_raw_data.py         # 数据清洗与规范化
│   └── maintenance/
│       └── rebuild_embedding_index.py
├── src/anti_fraud_explorer/
│   ├── web/
│   │   └── web.py                  # Flask 应用与路由
│   ├── agent/
│   │   ├── agent.py                # 核心 Agent：意图理解 -> 检索 -> 生成
│   │   ├── router.py               # 意图路由
│   │   ├── handlers.py             # 任务处理器（推荐/教案/转化等）
│   │   ├── formatting.py           # LLM 上下文格式化
│   │   ├── comparison.py           # 案例对比
│   │   └── models.py               # Agent 数据模型
│   ├── ai/
│   │   ├── client.py               # 模型调用封装
│   │   ├── spoken.py               # 口语化文案生成
│   │   ├── qa.py                   # 问答逻辑
│   │   └── context.py              # 上下文构建
│   ├── service/
│   │   ├── search.py               # 检索引擎（词法 + 混合）
│   │   ├── retriever.py            # 查询分析
│   │   ├── embeddings.py           # 向量嵌入
│   │   ├── tts.py                  # 语音合成
│   │   ├── http_client.py          # 统一 HTTP 客户端
│   │   ├── conversation.py         # 对话历史存储
│   │   └── item_cards.py           # 案例卡片渲染
│   ├── domain/
│   │   └── dataset.py              # 数据模型（KnowledgeBase / CaseItem）
│   ├── config.py                   # Pydantic Settings 配置
│   ├── prompts.py                  # 系统提示词
│   └── text.py                     # 文本规范化工具
├── static/                         # 前端静态资源
├── templates/                      # Jinja2 模板
├── tests/                          # 测试套件
└── docs/                           # 设计文档与规范
```

---

## 架构亮点

### 混合检索策略

检索层采用三级策略，兼顾速度与质量：

1. **词法评分**：标题精确匹配 > 子串匹配 > 分类匹配 > 摘要匹配 > token 匹配
2. **语义融合**（可选）：Embedding 相似度通过 RRF 与词法分融合
3. **拼音回退**：当词法标题匹配较弱时，通过 `pypinyin` 做同音/拼音容错

### Agent 多轮协作管道

```
用户提问
  -> 查询分析（改写/实体提取/场景识别）
  -> 首轮标题候选检索
  -> 模型决策（answer / search）
  -> [如需] 精查详情检索
  -> 最终答案生成 + 语音合成
```

模型每轮根据已有上下文判断是**直接回答**还是**补充检索**，最多执行 `MAX_SEARCH_ROUNDS_PER_TURN` 轮（默认 2 轮）。未配置 API Key 时自动降级为纯本地检索回答。

### 降级设计

| 组件 | 主方案 | 降级方案 |
|------|--------|----------|
| AI 问答 | LLM 生成 | 本地检索 + 模板回答 |
| 语音合成 | 火山引擎 TTS | OpenAI TTS -> 浏览器语音 |
| 语义检索 | Embedding API | 纯词法检索 |
| 意图规划 | Planner 模型 | 基于规则回退 |

---

## 数据模型

当前使用规范化案例结构（`schema_version: 4`），核心字段：

```json
{
  "case_id": "",
  "title": "",
  "summary": "",
  "ccl2023_category": "",
  "custom_subcategory": "",
  "risk_level": "",
  "entry_channels": [],
  "impersonated_identity": [],
  "false_belief": [],
  "key_methods": [],
  "target_assets": [],
  "fraud_stage": [],
  "risk_signals": "",
  "prevention_advice": "",
  "source_name": "",
  "victim_group": "",
  "tags": [],
  "involved_platforms": []
}
```

多值字段统一使用 JSON 数组。旧版兼容字段（`family`、`level`、`province` 等）已移除。

---

## 文档

- [检索设计](docs/检索设计.md)
- [数据采集与标注规范](docs/反诈数据采集与标注规范_简洁专业版_v1.2.docx)

---

## License

内部项目。
