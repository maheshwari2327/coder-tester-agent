"""Unit tests for agent.py using Mock LLMs to verify graph execution flow."""

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from agent import (
    MAX_FIXES,
    AgentState,
    build_graph,
    check_test_status,
    coder_node,
    debugger_node,
    run_tests_node,
    tester_node as agent_tester_node,
)


class MockLLM:
    """Mock LLM returning predefined responses in sequence."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):
        if not self.responses:
            raise RuntimeError("MockLLM: No more responses configured")
        resp = self.responses.pop(0)
        self.call_count += 1
        return AIMessage(content=resp)


def test_coder_node():
    mock_llm = MockLLM(["```python\ndef square(n):\n    return n * n\n```"])
    state = {
        "task": "square a number",
        "code": "",
        "tests": "",
        "passed": False,
        "output": "",
        "fixes": 0
    }
    result = coder_node(state, mock_llm)
    assert result["code"] == "def square(n):\n    return n * n"


def test_tester_node():
    mock_llm = MockLLM(["```python\nassert square(2) == 4\nassert square(0) == 0\nassert square(-3) == 9\nassert square(10) == 100\n```"])
    state = {
        "task": "square a number",
        "code": "def square(n):\n    return n * n",
        "tests": "",
        "passed": False,
        "output": "",
        "fixes": 0
    }
    result = agent_tester_node(state, mock_llm)
    assert "assert square(2) == 4" in result["tests"]


def test_run_tests_node_pass():
    state = {
        "task": "square a number",
        "code": "def square(n):\n    return n * n",
        "tests": "assert square(2) == 4\nassert square(0) == 0",
        "passed": False,
        "output": "",
        "fixes": 0
    }
    result = run_tests_node(state)
    assert result["passed"] is True
    assert "ALL TESTS PASSED" in result["output"]


def test_run_tests_node_fail():
    state = {
        "task": "square a number",
        "code": "def square(n):\n    return n + 1",
        "tests": "assert square(2) == 4",
        "passed": False,
        "output": "",
        "fixes": 0
    }
    result = run_tests_node(state)
    assert result["passed"] is False
    assert "AssertionError" in result["output"]


def test_debugger_node_two_blocks():
    mock_llm = MockLLM([
        "```python\ndef square(n):\n    return n * n\n```\n\n```python\nassert square(2) == 4\nassert square(3) == 9\n```"
    ])
    state = {
        "task": "square a number",
        "code": "def square(n):\n    return n + 1",
        "tests": "assert square(2) == 4",
        "passed": False,
        "output": "AssertionError",
        "fixes": 0
    }
    result = debugger_node(state, mock_llm)
    assert result["fixes"] == 1
    assert result["code"] == "def square(n):\n    return n * n"
    assert "assert square(3) == 9" in result["tests"]


def test_check_test_status():
    assert check_test_status({"passed": True, "fixes": 0}) == END
    assert check_test_status({"passed": False, "fixes": 0}) == "debugger"
    assert check_test_status({"passed": False, "fixes": MAX_FIXES}) == END


def test_graph_success_on_first_try():
    """Graph succeeds without triggering the debugger."""
    mock_llm = MockLLM([
        # Coder response
        "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
        # Tester response
        "```python\nassert is_even(2) is True\nassert is_even(3) is False\nassert is_even(0) is True\nassert is_even(-4) is True\n```"
    ])
    graph = build_graph(llm=mock_llm)
    result = graph.invoke({
        "task": "check if a number is even",
        "code": "",
        "tests": "",
        "passed": False,
        "output": "",
        "fixes": 0
    })

    assert result["passed"] is True
    assert result["fixes"] == 0
    assert "def is_even" in result["code"]
    assert mock_llm.call_count == 2


def test_graph_debugger_loop_and_recover():
    """Graph fails initially, debugger fixes it on first attempt, and it passes."""
    mock_llm = MockLLM([
        # Coder response (buggy)
        "```python\ndef is_even(n):\n    return n % 2 != 0\n```",
        # Tester response
        "```python\nassert is_even(2) is True\nassert is_even(3) is False\nassert is_even(0) is True\nassert is_even(-4) is True\n```",
        # Debugger response (fixed function & asserts)
        "```python\ndef is_even(n):\n    return n % 2 == 0\n```\n\n```python\nassert is_even(2) is True\nassert is_even(3) is False\nassert is_even(0) is True\nassert is_even(-4) is True\n```"
    ])
    graph = build_graph(llm=mock_llm)
    result = graph.invoke({
        "task": "check if a number is even",
        "code": "",
        "tests": "",
        "passed": False,
        "output": "",
        "fixes": 0
    })

    assert result["passed"] is True
    assert result["fixes"] == 1
    assert "return n % 2 == 0" in result["code"]
    assert mock_llm.call_count == 3


def test_graph_max_fixes_reached():
    """Graph fails repeatedly and exits when MAX_FIXES is reached."""
    mock_llm = MockLLM([
        # Coder response (buggy)
        "```python\ndef is_even(n):\n    return False\n```",
        # Tester response
        "```python\nassert is_even(2) is True\nassert is_even(3) is False\nassert is_even(0) is True\nassert is_even(-4) is True\n```",
        # Debugger 1
        "```python\ndef is_even(n):\n    return False\n```",
        # Debugger 2
        "```python\ndef is_even(n):\n    return False\n```",
        # Debugger 3
        "```python\ndef is_even(n):\n    return False\n```",
    ])
    graph = build_graph(llm=mock_llm)
    result = graph.invoke({
        "task": "check if a number is even",
        "code": "",
        "tests": "",
        "passed": False,
        "output": "",
        "fixes": 0
    })

    assert result["passed"] is False
    assert result["fixes"] == MAX_FIXES
