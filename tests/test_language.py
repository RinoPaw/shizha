from __future__ import annotations

from anti_fraud_explorer.language import (
    LANGUAGE_PROFILES,
    detect_locale,
    get_language_profile,
    is_chinese_locale,
    locale_from_provider_language,
    normalize_locale_hint,
)


def test_supported_scope_is_chinese_dialects_english_japanese_and_korean() -> None:
    assert set(LANGUAGE_PROFILES) == {
        "zh-CN",
        "yue-CN",
        "zh-CN-sichuan",
        "zh-CN-henan",
        "en-US",
        "ja-JP",
        "ko-KR",
    }


def test_locale_hints_are_canonicalized_without_accepting_unknown_values() -> None:
    assert normalize_locale_hint("zh_Hans") == "zh-CN"
    assert normalize_locale_hint("ZH-hk") == "yue-CN"
    assert normalize_locale_hint("zh_henan") == "zh-CN-henan"
    assert normalize_locale_hint("jp") == "ja-JP"
    assert normalize_locale_hint("fr-FR") is None
    assert normalize_locale_hint("auto") is None
    assert normalize_locale_hint("xx-Unknown") is None


def test_text_is_authoritative_for_language_and_dialect_detection() -> None:
    assert detect_locale("この無形文化遺産を紹介して", hint="en-US") == "ja-JP"
    assert detect_locale("佢嘕传统手艺有咩特色？", hint="zh-CN") == "yue-CN"
    assert detect_locale("这个手艺好巴适，咋个做的？") == "zh-CN-sichuan"
    assert detect_locale("恁看看这个手艺中不中？") == "zh-CN-henan"
    assert detect_locale("河南话呢") == "zh-CN-henan"
    assert detect_locale("Tell me about this heritage project", hint="zh-CN") == "en-US"
    assert detect_locale("곤곡이라는 전통 예술을 소개해 주세요") == "ko-KR"
    assert detect_locale("昆曲", hint="ja-JP") == "ja-JP"


def test_provider_language_tags_map_to_response_and_tts_profiles() -> None:
    expected = {
        "zh_cn": ("zh-CN", "zh-CN-XiaoshuangNeural"),
        "en": ("en-US", "en-US-JennyNeural"),
        "ja": ("ja-JP", "ja-JP-NanamiNeural"),
        "ko": ("ko-KR", "ko-KR-SunHiNeural"),
        "yue": ("yue-CN", "zh-HK-HiuMaanNeural"),
    }

    for provider_tag, (locale, voice) in expected.items():
        assert locale_from_provider_language(provider_tag) == locale
        assert get_language_profile(locale).tts_voice == voice

    assert locale_from_provider_language("unknown") is None


def test_chinese_dialect_profiles_share_chinese_asr_and_use_available_voices() -> None:
    assert is_chinese_locale("zh-CN")
    assert is_chinese_locale("yue-CN")
    assert is_chinese_locale("zh-CN-sichuan")
    assert is_chinese_locale("zh-CN-henan")
    assert not is_chinese_locale("en-US")
    assert get_language_profile("yue-CN").tts_voice == "zh-HK-HiuMaanNeural"
    assert get_language_profile("zh-CN-sichuan").tts_voice == "zh-CN-XiaoshuangNeural"
    assert get_language_profile("zh-CN-henan").tts_voice == "zh-CN-XiaoshuangNeural"
