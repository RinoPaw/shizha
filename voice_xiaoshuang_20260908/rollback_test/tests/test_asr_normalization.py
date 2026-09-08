from __future__ import annotations

from types import SimpleNamespace

from anti_fraud_explorer.asr_normalization import normalize_asr_final


def _kb() -> SimpleNamespace:
    return SimpleNamespace(
        items=[
            SimpleNamespace(
                title="刷单返利后要求垫资",
                category="刷单返利类",
                custom_subcategory="刷单返利",
                entry_channels=("微信",),
                key_methods=("小额返利", "虚假订单/任务"),
                involved_platforms=(),
                tags=("刷单返利",),
            ),
            SimpleNamespace(
                title="冒充公检法要求转入安全账户",
                category="冒充公检法及政府机关类",
                custom_subcategory="冒充公检法",
                entry_channels=("电话",),
                key_methods=("安全账户", "诱导转账"),
                involved_platforms=(),
                tags=("公检法",),
            ),
        ]
    )


def test_corrects_high_confidence_fraud_term_in_context() -> None:
    result = normalize_asr_final(
        "这是刷单反利骗局吗？",
        kb=_kb(),
        category="刷单返利类",
        asr_candidates=("这是刷单返利骗局吗", "这是刷单反利骗局吗"),
    )
    assert result.canonical_text == "这是刷单返利骗局吗？"
    assert result.spans
    assert result.spans[0].canonical == "刷单返利"


def test_exact_safety_terms_are_preserved() -> None:
    text = "对方让我共享屏幕，还要验证码。"
    result = normalize_asr_final(text, kb=_kb())
    assert result.canonical_text == text
    assert result.spans == ()


def test_ordinary_sentence_is_not_forced_to_a_case_title() -> None:
    text = "我刚刚睡醒，想先喝杯水。"
    result = normalize_asr_final(text, kb=_kb())
    assert result.canonical_text == text
    assert result.spans == ()


def test_nbest_does_not_override_a_phonetically_different_real_sentence() -> None:
    text = "今天返利活动结束了。"
    result = normalize_asr_final(
        text,
        kb=_kb(),
        asr_candidates=(text, "今天刷单返利活动结束了"),
    )
    assert result.canonical_text == text
    assert result.spans == ()


def test_non_chinese_transcript_skips_pinyin_normalization_even_with_context() -> None:
    result = normalize_asr_final(
        "刷单反利詐欺について教えてください。",
        kb=_kb(),
        category="刷单返利类",
        asr_candidates=("刷单返利詐欺",),
        language="ja",
    )

    assert result.raw_text == "刷单反利詐欺について教えてください。"
    assert result.canonical_text == result.raw_text
    assert result.spans == ()
