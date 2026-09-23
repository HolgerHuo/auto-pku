import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from notes import (  # noqa: E402
    _complete_plan,
    _parse_plan,
    _unit_material,
    is_cot_garbage,
    salvage_from_reasoning,
)


def test_reject_uppercase_planning_leak_after_markdown() -> None:
    text = """### 主题
- 一条有效笔记。

Need maybe include another example. Potential issue: this is uncertain.
"""

    assert is_cot_garbage(text) is True


def test_reject_planning_leak_near_start_of_long_response() -> None:
    text = (
        "### 主题\nNeed ensure exact terms.\n"
        + "- 后续笔记。\n" * 100
    )

    assert is_cot_garbage(text) is True


def test_reject_unenumerated_need_planning_line() -> None:
    text = "### 主题\n- 笔记。\nNeed decide whether to include this.\n"

    assert is_cot_garbage(text) is True


def test_accept_normal_study_notes() -> None:
    text = """### 游戏史
- 课程讨论街机的发展。
- 讲师用具体案例说明技术与文化之间的关系。
"""

    assert is_cot_garbage(text) is False


def test_unit_material_drops_contaminated_slide_but_keeps_speech() -> None:
    unit = {
        "parts": [
            (
                {},
                {
                    "t_start": 0,
                    "t_end": 10,
                    "slides": "Need decide how to describe this slide.",
                    "speech": "老师开始讲解课程内容。",
                },
            )
        ]
    }

    material = _unit_material(unit)

    assert "Need decide" not in material
    assert "老师开始讲解课程内容。" in material


def test_salvaged_reasoning_still_requires_garbage_validation() -> None:
    reasoning = """Planning first.
### 主题
- 第一条笔记内容，用于构成长篇结构化回答。
- 第二条笔记内容，用于构成长篇结构化回答。
Need check whether another fact belongs here.
- 第三条笔记内容，用于构成长篇结构化回答。
- 第四条笔记内容，用于构成长篇结构化回答。
- 第五条笔记内容，用于构成长篇结构化回答。
"""

    assert salvage_from_reasoning(reasoning) == ""


def test_parse_plan_extracts_json_object() -> None:
    text = (
        'Here is the plan: {"sections": [{"title": "街机时代", '
        '"description": "d", "segments": [0, 1]}, '
        '{"title": "游戏设计", "description": "d", "segments": [2]}]} '
        "Done."
    )

    plan = _parse_plan(text)

    assert len(plan) == 2
    assert plan[0]["title"] == "街机时代"
    assert plan[1]["segments"] == [2]


def test_complete_plan_accepts_full_coverage() -> None:
    plan = _complete_plan(
        [{"segments": [0, 1]}, {"segments": [2, 3, 4]}], 5)

    assert plan is not None
    assert [s["segments"] for s in plan] == [[0, 1], [2, 3, 4]]


def test_complete_plan_backfills_truncated_prefix() -> None:
    plan = _complete_plan([{"segments": [0, 1]}], 5)

    assert plan is not None
    assert plan[1]["segments"] == [2, 3, 4]


def test_complete_plan_rejects_gaps_dups_and_order() -> None:
    assert _complete_plan([{"segments": [0]}, {"segments": [2]}], 3) is None
    assert _complete_plan([{"segments": [0, 1]}, {"segments": [1]}], 3) is None
    assert _complete_plan([{"segments": [1, 2]}, {"segments": [0]}], 3) is None
    assert _complete_plan([{"segments": [2, 1]}], 3) is None
    assert _complete_plan([{"segments": "0,1"}], 3) is None


def test_parse_plan_salvages_truncated_json() -> None:
    text = (
        '{"sections": [{"title": "A", "description": "d", '
        '"segments": [0, 1]}, {"title": "B", "description": "d", '
        '"segments": [2, 3]}, {"title": "C", "description": "truncat'
    )

    plan = _parse_plan(text)

    assert [s["segments"] for s in plan] == [[0, 1], [2, 3]]
