"""Coder <-> Tester Multi-Agent System using LangChain and LangGraph.

Stack:
- LLM: Ollama llama3.2 (via langchain-ollama, temperature 0)
- Framework: LangGraph StateGraph
"""

import os
from typing import Any, Dict, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from tools import extract_all_code_blocks, run_python, strip_code_fences

# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

CODER_PROMPT = """You are a PYTHON CODER. Write ONE Python function that solves the task.
Rules: no input(), no print(), no example usage, no explanation.
Return ONLY the code inside one ```python block."""

TESTER_PROMPT = """You are a TESTER. Write 4 simple `assert` lines that test the function (call it by its name).
Rules: only assert lines, no imports, no explanation, no function definition.
Return ONLY the asserts inside one ```python block."""

DEBUGGER_PROMPT = """You are a DEBUGGER. The code failed its tests. You get the code, the tests and the error.
Either the code OR a test may be wrong (for example a wrong expected value) — decide which and fix it.
Return TWO ```python blocks and nothing else:
1) the full corrected function
2) the full list of assert lines (fix any assert that is itself wrong)"""

MAX_FIXES = 3


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    task: str
    code: str
    tests: str
    passed: bool
    output: str
    fixes: int


# ============================================================================
# LLM FACTORY
# ============================================================================

def get_llm(model: Optional[str] = None, temperature: float = 0.0) -> ChatOllama:
    """Get the Ollama Chat model with temperature 0."""
    target_model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=target_model, temperature=temperature, base_url=base_url)


# ============================================================================
# GRAPH NODES
# ============================================================================

def coder_node(state: AgentState, llm: Any) -> Dict[str, Any]:
    """Coder generates ONE Python function that solves the task."""
    messages = [
        SystemMessage(content=CODER_PROMPT),
        HumanMessage(content=f"Task: {state['task']}")
    ]
    response = llm.invoke(messages)
    code = strip_code_fences(response.content)
    return {"code": code}


def tester_node(state: AgentState, llm: Any) -> Dict[str, Any]:
    """Tester generates 4 simple assert lines testing the generated function."""
    messages = [
        SystemMessage(content=TESTER_PROMPT),
        HumanMessage(content=f"Task: {state['task']}\n\nFunction to test:\n{state['code']}")
    ]
    response = llm.invoke(messages)
    tests = strip_code_fences(response.content)
    return {"tests": tests}


def run_tests_node(state: AgentState) -> Dict[str, Any]:
    """Non-LLM node: runs code + tests + marker print, evaluates pass/fail."""
    full_script = f"{state['code']}\n\n{state['tests']}\n\nprint('ALL TESTS PASSED')"
    res = run_python(full_script, timeout=5)
    passed = bool(res.get("ok")) and ("ALL TESTS PASSED" in res.get("output", ""))
    return {
        "passed": passed,
        "output": res.get("output", "")
    }


def debugger_node(state: AgentState, llm: Any) -> Dict[str, Any]:
    """Debugger gets code, tests, and error output, then returns fixes."""
    messages = [
        SystemMessage(content=DEBUGGER_PROMPT),
        HumanMessage(
            content=(
                f"Task: {state['task']}\n\n"
                f"Code:\n{state['code']}\n\n"
                f"Tests:\n{state['tests']}\n\n"
                f"Error / Output:\n{state['output']}"
            )
        )
    ]
    response = llm.invoke(messages)
    blocks = extract_all_code_blocks(response.content)
    new_fixes = state.get("fixes", 0) + 1

    if len(blocks) >= 2:
        return {
            "code": blocks[0],
            "tests": blocks[1],
            "fixes": new_fixes
        }
    elif len(blocks) == 1:
        return {
            "code": blocks[0],
            "fixes": new_fixes
        }
    else:
        return {
            "code": strip_code_fences(response.content),
            "fixes": new_fixes
        }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def check_test_status(state: AgentState) -> str:
    """Route to END if passed or max fixes reached, else to debugger."""
    if state.get("passed", False) or state.get("fixes", 0) >= MAX_FIXES:
        return END
    return "debugger"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_graph(llm: Optional[Any] = None):
    """Build and compile the LangGraph StateGraph."""
    if llm is None:
        llm = get_llm()

    builder = StateGraph(AgentState)

    builder.add_node("coder", lambda state: coder_node(state, llm))
    builder.add_node("tester", lambda state: tester_node(state, llm))
    builder.add_node("run_tests", run_tests_node)
    builder.add_node("debugger", lambda state: debugger_node(state, llm))

    builder.add_edge(START, "coder")
    builder.add_edge("coder", "tester")
    builder.add_edge("tester", "run_tests")
    builder.add_conditional_edges(
        "run_tests",
        check_test_status,
        {
            END: END,
            "debugger": "debugger"
        }
    )
    builder.add_edge("debugger", "run_tests")

    return builder.compile()


if __name__ == "__main__":
    from main import main
    main()
