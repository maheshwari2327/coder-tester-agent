"""Tests for tools.py."""

from tools import strip_code_fences, extract_all_code_blocks, run_python


def test_strip_code_fences_single_python():
    text = "```python\ndef add(a, b):\n    return a + b\n```"
    expected = "def add(a, b):\n    return a + b"
    assert strip_code_fences(text) == expected


def test_strip_code_fences_single_generic():
    text = "```\nassert add(1, 2) == 3\n```"
    expected = "assert add(1, 2) == 3"
    assert strip_code_fences(text) == expected


def test_strip_code_fences_raw():
    text = "def add(a, b):\n    return a + b"
    assert strip_code_fences(text) == text


def test_extract_all_code_blocks():
    text = (
        "Here is the function:\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "Here are the asserts:\n"
        "```python\nassert add(1, 2) == 3\nassert add(0, 0) == 0\n```"
    )
    blocks = extract_all_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0] == "def add(a, b):\n    return a + b"
    assert "assert add(1, 2) == 3" in blocks[1]


def test_run_python_success():
    code = "x = 10\ny = 20\nprint(x + y)\nprint('ALL TESTS PASSED')"
    res = run_python(code, timeout=5)
    assert res["ok"] is True
    assert "ALL TESTS PASSED" in res["output"]


def test_run_python_assertion_failure():
    code = "assert 1 == 2, 'Value mismatch'"
    res = run_python(code, timeout=5)
    assert res["ok"] is False
    assert "AssertionError" in res["output"]


def test_run_python_timeout():
    code = "import time\ntime.sleep(2)"
    res = run_python(code, timeout=1)
    assert res["ok"] is False
    assert "TimeoutExpired" in res["output"]
