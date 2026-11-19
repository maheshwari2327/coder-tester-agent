"""Streamlit Web UI for Coder-Tester Multi-Agent System."""

import os
import streamlit as st
from langchain_core.messages import AIMessage

from agent import MAX_FIXES, build_graph, get_llm


# Set up page configuration
st.set_page_config(
    page_title="Coder-Tester Multi-Agent",
    page_icon="🤖",
    layout="wide"
)

# Student Safety Warning Banner
st.warning(
    "⚠️ **Classroom Safety Notice:** This agent executes automatically generated Python code "
    "locally via `subprocess`. Running untrusted AI-generated code is only safe for classroom demos "
    "and local sandboxed testing.",
    icon="⚠️"
)

# App Title & Description
st.title("🤖 Coder <-> Tester Multi-Agent System")
st.caption(
    "Autonomous code generation, test generation, test execution, and self-debugging loop "
    "built with **LangChain**, **LangGraph**, and **Ollama**."
)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Agent Settings")
    ollama_model = st.text_input("Ollama Model", value="llama3.2")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    
    st.divider()
    demo_mode = st.checkbox(
        "Simulation / Demo Mode (Mock LLM)",
        value=False,
        help="Simulate the agent flow with pre-scripted responses to test the UI without needing a running Ollama server."
    )

    st.divider()
    st.subheader("💡 Example Tasks")
    example_tasks = [
        "Write a function to check if a word is an anagram of another",
        "Write a function to find the maximum sum of a contiguous subarray (Kadane's)",
        "Write a function to validate if a string has balanced parentheses",
        "Write a function to count the frequency of each word in a string"
    ]
    for ex in example_tasks:
        if st.button(ex, use_container_width=True):
            st.session_state["task_input"] = ex

# Mock LLM for Simulation Mode
class DemoSimulationLLM:
    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        # Step 1: Buggy Coder
        if self.call_count == 1:
            return AIMessage(
                content="```python\ndef is_anagram(s1: str, s2: str) -> bool:\n    # Buggy initial version\n    return sorted(s1) == sorted(s2) and len(s1) > 0\n```"
            )
        # Step 2: Tester
        elif self.call_count == 2:
            return AIMessage(
                content="```python\nassert is_anagram('listen', 'silent') is True\nassert is_anagram('hello', 'world') is False\nassert is_anagram('', '') is True\nassert is_anagram('rat', 'car') is False\n```"
            )
        # Step 3: Debugger (fixes the empty string edge case)
        else:
            return AIMessage(
                content=(
                    "```python\ndef is_anagram(s1: str, s2: str) -> bool:\n    return sorted(s1) == sorted(s2)\n```\n\n"
                    "```python\nassert is_anagram('listen', 'silent') is True\nassert is_anagram('hello', 'world') is False\nassert is_anagram('', '') is True\nassert is_anagram('rat', 'car') is False\n```"
                )
            )

# Main UI Input
task_input = st.text_input(
    "Enter your coding task:",
    value=st.session_state.get("task_input", ""),
    placeholder="e.g. Write a function to check if a number is prime"
)

col1, col2 = st.columns([1, 5])
with col1:
    start_btn = st.button("🚀 Run Agent", type="primary", use_container_width=True)

if start_btn:
    if not task_input.strip():
        st.error("Please enter a coding task first.")
    else:
        # Configure LLM
        os.environ["OLLAMA_MODEL"] = ollama_model
        os.environ["OLLAMA_BASE_URL"] = ollama_url

        if demo_mode:
            llm = DemoSimulationLLM()
            st.info("ℹ️ Running in Simulation Mode (using simulated multi-step agent responses).")
        else:
            llm = get_llm(model=ollama_model)

        try:
            graph = build_graph(llm=llm)
        except Exception as e:
            st.error(f"Failed to initialize graph: {e}")
            st.stop()

        initial_state = {
            "task": task_input,
            "code": "",
            "tests": "",
            "passed": False,
            "output": "",
            "fixes": 0
        }

        final_state = dict(initial_state)

        st.subheader("🔄 Multi-Agent Execution Progress")
        progress_container = st.container()

        with st.spinner("Multi-agent system is working..."):
            try:
                for event in graph.stream(initial_state, stream_mode="updates"):
                    for node_name, node_update in event.items():
                        final_state.update(node_update)

                        with progress_container:
                            if node_name == "coder":
                                with st.expander("📝 Node: `coder` — Generated Function", expanded=True):
                                    st.code(node_update.get("code", ""), language="python")

                            elif node_name == "tester":
                                with st.expander("🧪 Node: `tester` — Generated 4 Assert Tests", expanded=True):
                                    st.code(node_update.get("tests", ""), language="python")

                            elif node_name == "run_tests":
                                passed = node_update.get("passed", False)
                                output = node_update.get("output", "").strip()
                                with st.expander(
                                    f"⚙️ Node: `run_tests` — {'✅ PASSED' if passed else '❌ FAILED'}",
                                    expanded=True
                                ):
                                    if passed:
                                        st.success("All test assertions passed successfully!")
                                    else:
                                        st.error("Tests failed with error output:")
                                    if output:
                                        st.code(output, language="text")

                            elif node_name == "debugger":
                                fixes = node_update.get("fixes", final_state.get("fixes", 0))
                                with st.expander(f"🔧 Node: `debugger` — Fix Attempt {fixes}/{MAX_FIXES}", expanded=True):
                                    st.write("**Corrected Function:**")
                                    st.code(node_update.get("code", ""), language="python")
                                    if "tests" in node_update:
                                        st.write("**Corrected Tests:**")
                                        st.code(node_update.get("tests", ""), language="python")

            except Exception as e:
                st.error(f"Error during agent execution: {e}")
                if "11434" in str(e) or "connect" in str(e).lower():
                    st.warning(
                        "Could not connect to Ollama on http://localhost:11434. "
                        "Make sure Ollama is running (`ollama serve`), or enable 'Simulation / Demo Mode' in the sidebar."
                    )

        # Final Summary
        st.divider()
        st.subheader("📊 Final Result")

        if final_state.get("passed"):
            st.success(f"🎉 **Status: PASSED** (Completed with {final_state.get('fixes', 0)} fixes)")
        else:
            st.error(f"💥 **Status: FAILED** (Reached maximum fix limit: {final_state.get('fixes', 0)}/{MAX_FIXES})")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("### 📄 Final Function Code")
            st.code(final_state.get("code", "").strip() or "# No code generated", language="python")

        with res_col2:
            st.markdown("### 🧪 Final Tests")
            st.code(final_state.get("tests", "").strip() or "# No tests generated", language="python")
