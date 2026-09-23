import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from asr import _clean_text, _strip_context_echo  # noqa: E402


def test_strip_repeated_prompt_echo() -> None:
    echoed = "电子游戏如何改变世界 Key terms (spell exactly): 电子游戏, 游戏"
    prompt = f"Lecture topic: {echoed}"
    text = echoed * 3 + "各位同学，大家好。"

    cleaned, triggered = _strip_context_echo(text, prompt, "电子游戏, 游戏")

    assert triggered is True
    assert cleaned == "各位同学，大家好。"


def test_preserve_lecture_text_without_context_echo() -> None:
    text = "老师解释了电子游戏如何改变世界。"

    cleaned, triggered = _strip_context_echo(
        text,
        "电子游戏如何改变世界 Key terms (spell exactly): 电子游戏, 游戏",
        "电子游戏, 游戏",
    )

    assert triggered is False
    assert cleaned == text


def test_strip_key_term_list_echo() -> None:
    prompt = (
        "Lecture topic: 导论完结篇 "
        "Key terms (spell exactly): Nintendo, Game design, 导论, 完结篇"
    )

    cleaned, triggered = _strip_context_echo(
        "课程到这里结束。Nintendo, Game design, 导论, 完结篇。",
        prompt,
    )

    assert triggered is True
    assert cleaned == "课程到这里结束。。"


def test_strip_bare_key_terms_banner() -> None:
    cleaned, triggered = _strip_context_echo(
        "开场。Key terms (spell exactly): 好，开始上课。",
        "Lecture topic: 人类沟通",
    )

    assert triggered is True
    assert cleaned == "开场。 好，开始上课。"


def test_strip_repeated_topic_banner_tail() -> None:
    text = (
        "课堂结束。"
        "Lecture topic: 意大利语 清辅音, 浊辅音, "
        "Lecture topic: 意大利语 清辅音, 浊辅音, "
        "Lecture topic: 意大利语 清辅音, 浊辅音"
    )

    cleaned, triggered = _strip_context_echo(
        text,
        "Lecture topic: 意大利语 Key terms (spell exactly): 清辅音, 浊辅音",
        "清辅音, 浊辅音",
    )

    assert triggered is True
    assert cleaned == "课堂结束。"


def test_collapse_ascii_period_hallucination_loop() -> None:
    cleaned, triggered = _clean_text("Shine on. " * 100 + "Lezione iniziata.")

    assert triggered is True
    assert cleaned.count("Shine on.") == 3
    assert cleaned.endswith("Lezione iniziata.")


def test_collapse_alternating_sentence_loop_and_character_run() -> None:
    text = ("I'm so sad. I can't breathe. " * 100) + ("哒" * 100)

    cleaned, triggered = _clean_text(text)

    assert triggered is True
    assert cleaned.count("I'm so sad.") == 3
    assert cleaned.count("I can't breathe.") == 3
    assert "哒" * 4 not in cleaned


def test_collapse_comma_separated_token_loop() -> None:
    cleaned, triggered = _clean_text("开始。" + "ciao, " * 100 + "继续。")

    assert triggered is True
    assert cleaned.count("ciao") == 3
    assert cleaned.endswith("继续。")


def test_collapse_comma_separated_phrase_loop() -> None:
    cleaned, triggered = _clean_text("开场。" + "se c'e un problema, " * 50 + "继续。")

    assert triggered is True
    assert cleaned.count("se c'e un problema") == 3
    assert cleaned.endswith("继续。")


def test_cap_unpunctuated_hallucination_tail() -> None:
    cleaned, triggered = _clean_text("课堂内容" + "abcdef" * 1000)

    assert triggered is True
    assert len(cleaned) == 2000
    assert cleaned.startswith("课堂内容")
