"""Application paths and environment-backed settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 配置模型
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """应用配置，自动从 .env 文件和系统环境变量加载。

    * 环境变量名通过 alias 映射到 Python 字段名。
    * 系统环境变量的优先级高于 .env 文件。
    * 相对路径自动转换为项目根目录下的绝对路径。
    * 布尔字段可直接使用 1/0 / true/false / yes/no 等表示。
    """

    model_config = {
        "env_file": PROJECT_ROOT / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }

    # ---- 基础服务 ----------------------------------------------------------
    dataset_path: Path = Field(default=Path("data/processed/case_items.json"), alias="DATASET_PATH")
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=5070, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # ---- AI 大模型 ---------------------------------------------------------
    ai_api_key: str = Field(default="", alias="AI_API_KEY")
    ai_base_url: str = Field(default="https://api.deepseek.com", alias="AI_BASE_URL")
    ai_model: str = Field(default="deepseek-v4-flash", alias="AI_MODEL")
    ai_timeout: int = Field(default=60, alias="AI_TIMEOUT")
    ai_max_context_chars: int = Field(default=5200, alias="AI_MAX_CONTEXT_CHARS")
    ai_agent_planner: bool = Field(default=True, alias="AI_AGENT_PLANNER")

    # ---- 向量嵌入与检索 ----------------------------------------------------
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field(
        default="https://api.vectorengine.ai/v1", alias="EMBEDDING_BASE_URL"
    )
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_timeout: int = Field(default=60, alias="EMBEDDING_TIMEOUT")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")
    embedding_workers: int = Field(default=6, alias="EMBEDDING_WORKERS")
    embedding_request_timeout: float = Field(default=5.0, alias="EMBEDDING_REQUEST_TIMEOUT")
    embedding_max_retries: int = Field(default=4, alias="EMBEDDING_MAX_RETRIES")
    embedding_retry_backoff: float = Field(default=3.0, alias="EMBEDDING_RETRY_BACKOFF")
    embedding_request_delay: float = Field(default=0.0, alias="EMBEDDING_REQUEST_DELAY")
    embedding_index_path: Path = Field(
        default=Path("data/embeddings/case_embeddings.json"),
        alias="EMBEDDING_INDEX_PATH",
    )
    embedding_text_max_chars: int = Field(default=1400, alias="EMBEDDING_TEXT_MAX_CHARS")
    embedding_min_score: float = Field(default=0.15, alias="EMBEDDING_MIN_SCORE")
    search_use_embedding: bool = Field(default=False, alias="SEARCH_USE_EMBEDDING")

    # ---- TTS 语音合成（火山引擎） -------------------------------------------
    volc_tts_enabled: bool = Field(default=True, alias="VOLC_TTS_ENABLED")
    openai_tts_enabled: bool = Field(default=False, alias="OPENAI_TTS_ENABLED")
    volc_tts_api_version: str = Field(default="auto", alias="VOLC_TTS_API_VERSION")
    volc_tts_endpoint: str = Field(
        default="https://openspeech.bytedance.com/api/v1/tts",
        alias="VOLC_TTS_ENDPOINT",
    )
    volc_tts_v3_endpoint: str = Field(
        default="https://openspeech.bytedance.com/api/v3/tts/unidirectional",
        alias="VOLC_TTS_V3_ENDPOINT",
    )
    volc_tts_api_key: str = Field(default="", alias="VOLC_TTS_API_KEY")
    volc_tts_app_id: str = Field(default="", alias="VOLC_TTS_APP_ID")
    volc_tts_access_token: str = Field(default="", alias="VOLC_TTS_ACCESS_TOKEN")
    volc_tts_cluster: str = Field(default="volcano_tts", alias="VOLC_TTS_CLUSTER")
    volc_tts_resource_id: str = Field(
        default="volc.service_type.10029", alias="VOLC_TTS_RESOURCE_ID"
    )
    volc_tts_voice_type: str = Field(
        default="zh_male_tiancaitongsheng_mars_bigtts",
        alias="VOLC_TTS_VOICE_TYPE",
    )
    volc_tts_emotion: str = Field(default="", alias="VOLC_TTS_EMOTION")
    volc_tts_emotion_scale: int = Field(default=4, alias="VOLC_TTS_EMOTION_SCALE")
    volc_tts_encoding: str = Field(default="mp3", alias="VOLC_TTS_ENCODING")
    volc_tts_rate: int = Field(default=24000, alias="VOLC_TTS_RATE")
    volc_tts_speed_ratio: float = Field(default=1.0, alias="VOLC_TTS_SPEED_RATIO")
    volc_tts_volume_ratio: float = Field(default=1.0, alias="VOLC_TTS_VOLUME_RATIO")
    volc_tts_pitch_ratio: float = Field(default=1.0, alias="VOLC_TTS_PITCH_RATIO")
    volc_tts_timeout: float = Field(default=20.0, alias="VOLC_TTS_TIMEOUT")
    volc_tts_max_chunk_bytes: int = Field(default=900, alias="VOLC_TTS_MAX_CHUNK_BYTES")
    tts_cache_dir: Path = Field(default=Path("tmp/tts"), alias="TTS_CACHE_DIR")

    # ---- ASR 语音识别（火山引擎） -------------------------------------------
    volc_asr_enabled: bool = Field(default=True, alias="VOLC_ASR_ENABLED")
    volc_asr_endpoint: str = Field(
        default="https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
        alias="VOLC_ASR_ENDPOINT",
    )
    volc_asr_api_key: str = Field(default="", alias="VOLC_ASR_API_KEY")
    volc_asr_app_id: str = Field(default="", alias="VOLC_ASR_APP_ID")
    volc_asr_access_token: str = Field(default="", alias="VOLC_ASR_ACCESS_TOKEN")
    volc_asr_resource_id: str = Field(
        default="volc.bigasr.auc_turbo", alias="VOLC_ASR_RESOURCE_ID"
    )
    volc_asr_timeout: float = Field(default=30.0, alias="VOLC_ASR_TIMEOUT")

    # ---- 路径处理：相对路径自动转为项目根目录下的绝对路径 ----------------
    @field_validator(
        "dataset_path",
        "embedding_index_path",
        "tts_cache_dir",
    )
    @classmethod
    def resolve_to_absolute(cls, v: Path) -> Path:
        """若为相对路径，则拼接到 PROJECT_ROOT 上。"""
        if not v.is_absolute():
            return PROJECT_ROOT / v
        return v


# ---------------------------------------------------------------------------
# 全局配置实例 —— 其他模块通过 `from settings import settings` 使用
# ---------------------------------------------------------------------------
settings = Settings()
